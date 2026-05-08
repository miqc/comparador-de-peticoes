from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import torch
import io

app = FastAPI()

# Carrega apenas o modelo de similaridade (Embeddings)
model_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extrair_e_validar_peticao(texto_completo, nome_arquivo):
    """
    Aplica regras heurísticas para garantir que o documento é uma petição
    e extrai apenas as seções relevantes (Fatos e Pedidos).
    """
    texto_maiusculo = texto_completo.upper()

    # 1. Checagem de Cabeçalho (Opcional, mas muito forte)
    # Procura termos jurídicos essenciais nas primeiras 2000 letras do documento
    termos_chave_peticao = ["EXCELENTÍSSIMO", "JUIZ DE DIREITO", "VARA", "COMARCA", "AÇÃO", "JUIZADO ESPECIAL"]
    cabecalho = texto_maiusculo[:2000] 
    
    tem_caracteristica_juridica = any(termo in cabecalho for termo in termos_chave_peticao)

    # 2. Busca pelas seções principais
    inicio_fatos = texto_maiusculo.find("DOS FATOS")
    inicio_pedidos = texto_maiusculo.find("DOS PEDIDOS")

    # REGRA DE CORTE (O Porteiro Heurístico)
    # Se não tem o vocabulário de cabeçalho e não tem as seções, rejeita imediatamente.
    if not tem_caracteristica_juridica and inicio_fatos == -1 and inicio_pedidos == -1:
        raise ValueError(f"O arquivo '{nome_arquivo}' não possui a estrutura ou os termos de uma petição legal válida.")

    # 3. Extração (Se passou pelo porteiro)
    secoes = []
    if inicio_fatos != -1:
        proxima_secao_apos_fatos = texto_maiusculo.find("DO DIREITO", inicio_fatos)
        if proxima_secao_apos_fatos == -1:
            proxima_secao_apos_fatos = inicio_pedidos if inicio_pedidos > inicio_fatos else len(texto_maiusculo)
        
        secoes.append(texto_completo[inicio_fatos:proxima_secao_apos_fatos])

    if inicio_pedidos != -1:
        secoes.append(texto_completo[inicio_pedidos:])
        
    # Se a petição for atípica/curta e só tiver passado pelo cabeçalho, devolve ela toda
    return "\n".join(secoes) if secoes else texto_completo

def dividir_em_blocos(texto, limite_palavras=300):
    palavras = texto.split()
    if not palavras:
        return [""]
        
    blocos = []
    for i in range(0, len(palavras), limite_palavras):
        bloco = " ".join(palavras[i:i+limite_palavras])
        blocos.append(bloco)
    return blocos

def gerar_embedding_documento(texto_relevante):
    blocos = dividir_em_blocos(texto_relevante)
    embeddings_blocos = model_embeddings.encode(blocos, convert_to_tensor=True)
    embedding_final = torch.mean(embeddings_blocos, dim=0, keepdim=True)
    return embedding_final

def ler_pdf_e_validar(arquivo_pdf: UploadFile, nome_arquivo: str):
    conteudo = arquivo_pdf.file.read()
    leitor_pdf = PdfReader(io.BytesIO(conteudo))
    
    texto_completo = ""
    for page in leitor_pdf.pages:
        texto_completo += page.extract_text() or ""
    
    # Extrai e valida ao mesmo tempo usando regras leves (Regex/Find)
    texto_relevante = extrair_e_validar_peticao(texto_completo, nome_arquivo)
    
    return texto_relevante.strip()

@app.post("/comparar-peticoes/")
async def comparar_peticoes_api(peticao1: UploadFile = File(...), peticao2: UploadFile = File(...)):
    try:
        texto1 = ler_pdf_e_validar(peticao1, "Petição 1")
        texto2 = ler_pdf_e_validar(peticao2, "Petição 2")

        if not texto1 or not texto2:
            return {"erro": "Um ou ambos os PDFs não possuem texto legível."}

        emb1 = gerar_embedding_documento(texto1)
        emb2 = gerar_embedding_documento(texto2)

        similaridade = util.cos_sim(emb1, emb2)
        percentual = similaridade.item() * 100

        if percentual >= 65.0:
            classificacao = "Mesmo assunto"
        else:
            classificacao = "Sem relação"

        resultado = {
            "classificacao": classificacao,
            "similaridade_percentual": f"{percentual:.2f}%",
        }
            
        return resultado

    except ValueError as ve:
        # Retorna o erro de validação heurística para o frontend exibir
        return {
            "erro_validacao": str(ve),
            "status": "documento_invalido"
        }
    except Exception as e:
        return {"erro_interno": str(e)}