from fastapi import FastAPI, File, UploadFile
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import uvicorn
import torch
import io
import sqlite3
import json
import re

app = FastAPI()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

DB_PATH = "peticoes_juridicas.db"

# --- CONFIGURAÇÃO INICIAL DO BANCO ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS peticoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_arquivo TEXT,
            merito TEXT,
            vetor TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- AUXILIARES DE PLN (MÉRITO E VALIDAÇÃO) ---
def extrair_e_validar_merito(texto_completo, nome_arquivo):
    texto_maiusculo = texto_completo.upper()
    pontuacao = 0
    
    # Evidências Jurídicas [cite: 567, 780]
    evidencias = {
        r'\bEXCELENTÍSSIMO\b': 2, r'\bDOUTO JUÍZO\b': 2, r'\bVARA\b': 1,
        r'\bCOMARCA\b': 1, r'\bAÇÃO ORDINÁRIA\b': 2, r'\bPEDE DEFERIMENTO\b': 2
    }

    for padrao, pontos in evidencias.items():
        if re.search(padrao, texto_maiusculo):
            pontuacao += pontos

    inicio_fatos = texto_maiusculo.find("DOS FATOS")
    if inicio_fatos != -1: pontuacao += 3

    if pontuacao < 4:
        raise ValueError(f"Arquivo '{nome_arquivo}' reprovado no score jurídico.")

    # Foco no Mérito para evitar similaridade artificial [cite: 693, 822]
    if inicio_fatos != -1:
        fim_fatos = texto_maiusculo.find("DO DIREITO", inicio_fatos)
        if fim_fatos == -1:
            fim_fatos = texto_maiusculo.find("DOS PEDIDOS", inicio_fatos)
        return texto_completo[inicio_fatos:fim_fatos].strip()
    
    return texto_completo[:4000].strip()

# --- ENDPOINTS ---

@app.post("/armazenar-peticao/")
async def armazenar_peticao(arquivo: UploadFile = File(...)):
    """Lê o PDF, valida, gera o vetor e salva no SQLite."""
    try:
        conteudo = await arquivo.read()
        leitor = PdfReader(io.BytesIO(conteudo))
        texto_completo = "".join([page.extract_text() or "" for page in leitor.pages])
        
        merito = extrair_e_validar_merito(texto_completo, arquivo.filename)
        embedding = model.encode(merito, convert_to_tensor=True)
        
        # Converte o tensor para lista e depois para string JSON para salvar no SQLite
        vetor_json = json.dumps(embedding.tolist())

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO peticoes (nome_arquivo, merito, vetor) VALUES (?, ?, ?)", 
                       (arquivo.filename, merito, vetor_json))
        conn.commit()
        conn.close()

        return {"status": "sucesso", "arquivo": arquivo.filename, "mensagem": "Petição indexada no banco."}
    except Exception as e:
        return {"erro": str(e)}

@app.post("/comparar-com-banco/")
async def comparar_com_banco(arquivo_novo: UploadFile = File(...)):
    """Compara a petição enviada contra TODAS as petições do banco de dados."""
    try:
        # 1. Processar a nova petição
        conteudo = await arquivo_novo.read()
        leitor = PdfReader(io.BytesIO(conteudo))
        texto_novo = "".join([page.extract_text() or "" for page in leitor.pages])
        merito_novo = extrair_e_validar_merito(texto_novo, arquivo_novo.filename)
        emb_novo = model.encode(merito_novo, convert_to_tensor=True)

        # 2. Buscar todos os vetores do banco
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT nome_arquivo, vetor FROM peticoes")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            return {"aviso": "O banco de dados está vazio."}

        # 3. Preparar a comparação em massa
        nomes = [r[0] for r in registros]
        # Converte strings JSON de volta para Tensores do PyTorch
        vetores_base = torch.tensor([json.loads(r[1]) for r in registros])

        # 4. Cálculo de Similaridade de Cosseno (Matriz vs Vetor)
        # S_C = (A . B) / (||A|| ||B||)
        similaridades = util.cos_sim(emb_novo, vetores_base)[0]
        
        # Pega o melhor resultado
        maior_score, indice = torch.max(similaridades, dim=0)
        percentual = maior_score.item() * 100

        return {
            "classificacao": "Mesmo assunto" if percentual >= 65 else "Assunto diferente",
            "similaridade": f"{percentual:.2f}%",
            "melhor_match": nomes[indice.item()],
            "analisados": len(nomes)
        }

    except Exception as e:
        return {"erro": str(e)}
    
# --- NOVOS ENDPOINTS (CRUD) ---

@app.get("/peticoes/")
async def listar_peticoes():
    """Retorna uma lista de todas as petições indexadas no banco."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Selecionamos apenas id e nome para não sobrecarregar a resposta com vetores longos
        cursor.execute("SELECT id, nome_arquivo FROM peticoes")
        registros = cursor.fetchall()
        conn.close()

        lista = [{"id": r[0], "nome_arquivo": r[1]} for r in registros]
        return {"total": len(lista), "peticoes": lista}
    except Exception as e:
        return {"erro": str(e)}

@app.get("/peticoes/{id}")
async def obter_peticao(id: int):
    """Busca os detalhes de uma petição específica pelo ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, nome_arquivo, merito FROM peticoes WHERE id = ?", (id,))
        registro = cursor.fetchone()
        conn.close()

        if not registro:
            return {"erro": "Petição não encontrada."}

        return {
            "id": registro[0],
            "nome_arquivo": registro[1],
            "merito_extraido": registro[2]
        }
    except Exception as e:
        return {"erro": str(e)}

@app.delete("/peticoes/{id}")
async def deletar_peticao(id: int):
    """Remove uma petição da base de dados pelo ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verifica se existe antes de deletar
        cursor.execute("SELECT id FROM peticoes WHERE id = ?", (id,))
        if not cursor.fetchone():
            conn.close()
            return {"erro": "ID não encontrado no banco de dados."}

        cursor.execute("DELETE FROM peticoes WHERE id = ?", (id,))
        conn.commit()
        conn.close()

        return {"status": "sucesso", "mensagem": f"Petição ID {id} removida com sucesso."}
    except Exception as e:
        return {"erro": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)