from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import torch
import io
import uvicorn
import re

app = FastAPI()

# Carrega o modelo de similaridade
model_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extrair_merito_peticao(texto_completo, nome_arquivo):
    """
    Valida se é uma petição e extrai ESTRITAMENTE a seção de fatos (o mérito),
    eliminando o ruído dos pedidos e jargões processuais genéricos.
    """
    texto_maiusculo = texto_completo.upper()
    pontuacao = 0
    
    # 1. Validação por Score (Porteiro Heurístico)
    evidencias = {
        r'\bEXCELENTÍSSIMO\b': 2,
        r'\bDOUTO JUÍZO\b': 2,
        r'\bVARA\b': 1,
        r'\bCOMARCA\b': 1,
        r'\bJUIZADO ESPECIAL\b': 1,
        r'\bAÇÃO ORDINÁRIA\b': 2,
        r'\bAÇÃO DE\b': 1,
        r'\bFAZENDA PÚBLICA\b': 1,
        r'\bQUALIFICADO\b': 1,
        r'\bPEDE DEFERIMENTO\b': 2
    }

    for padrao, pontos in evidencias.items():
        if re.search(padrao, texto_maiusculo):
            pontuacao += pontos

    inicio_fatos = texto_maiusculo.find("DOS FATOS")
    inicio_pedidos = texto_maiusculo.find("DOS PEDIDOS")
    
    if inicio_fatos != -1: pontuacao += 3
    if inicio_pedidos != -1: pontuacao += 3

    if pontuacao < 4:
        raise ValueError(f"O arquivo '{nome_arquivo}' não possui elementos suficientes de uma petição legal válida (Score: {pontuacao}).")

    # 2. Extração Focada no Assunto (MÉRITO)
    # Ignoramos os pedidos, pois eles contêm muitos termos genéricos (citação, honorários, etc)
    if inicio_fatos != -1:
        # O fim dos fatos geralmente é onde começa 'DO DIREITO' ou 'DOS PEDIDOS'
        fim_fatos = texto_maiusculo.find("DO DIREITO", inicio_fatos)
        if fim_fatos == -1:
            fim_fatos = inicio_pedidos if inicio_pedidos > inicio_fatos else len(texto_maiusculo)
        
        texto_fatos = texto_completo[inicio_fatos:fim_fatos].strip()
        
        # Se os fatos forem muito curtos (menos de 100 caracteres), pegamos um pouco mais de contexto
        if len(texto_fatos) > 100:
            return texto_fatos

    # Caso não ache 'DOS FATOS' explicitamente ou seja muito curto, pegamos o início 
    # (onde costuma haver a qualificação e o nome da ação) até um limite seguro.
    return texto_completo[:4000].strip()

def dividir_em_blocos(texto, limite_palavras=300):
    palavras = texto.split()
    if not palavras:
        return [""]
    blocos = [" ".join(palavras[i:i+limite_palavras]) for i in range(0, len(palavras), limite_palavras)]
    return blocos

def gerar_embedding_documento(texto_relevante):
    blocos = dividir_em_blocos(texto_relevante)
    embeddings_blocos = model_embeddings.encode(blocos, convert_to_tensor=True)
    embedding_final = torch.mean(embeddings_blocos, dim=0, keepdim=True)
    return embedding_final

def ler_pdf_e_extrair_merito(arquivo_pdf: UploadFile):
    conteudo = arquivo_pdf.file.read()
    leitor_pdf = PdfReader(io.BytesIO(conteudo))
    
    texto_completo = ""
    for page in leitor_pdf.pages:
        texto_completo += page.extract_text() or ""
    
    # Extrai especificamente a parte que define o ASSUNTO
    return extrair_merito_peticao(texto_completo, arquivo_pdf.filename)

@app.post("/comparar-peticoes/")
async def comparar_peticoes_api(peticao1: UploadFile = File(...), peticao2: UploadFile = File(...)):
    try:
        # Extraímos apenas o MÉRITO (Fatos) de cada uma
        merito1 = ler_pdf_e_extrair_merito(peticao1)
        merito2 = ler_pdf_e_extrair_merito(peticao2)

        if not merito1 or not merito2:
            return {"erro": "Conteúdo insuficiente para análise em um dos documentos."}

        # Gera embeddings baseados apenas no assunto real
        emb1 = gerar_embedding_documento(merito1)
        emb2 = gerar_embedding_documento(merito2)

        similaridade = util.cos_sim(emb1, emb2)
        percentual = similaridade.item() * 100

        # Com o ruído removido, 65% torna-se um limite muito mais confiável
        classificacao = "Mesmo assunto" if percentual >= 65.0 else "Sem relação"

        return {
            "classificacao": classificacao,
            "similaridade_percentual": f"{percentual:.2f}%"
        }

    except ValueError as ve:
        return {"erro_validacao": str(ve), "status": "documento_invalido"}
    except Exception as e:
        return {"erro_interno": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)