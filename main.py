from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import uvicorn
import torch
import io
import sqlite3
import json
import re
import os
import shutil

app = FastAPI()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# CORS para o frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração de diretórios
DB_PATH = "peticoes_juridicas.db"
UPLOAD_DIR = "arquivos_cartorio"
os.makedirs(f"{UPLOAD_DIR}/peticoes", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/despachos", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS peticoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_peticao TEXT,
            merito TEXT,
            vetor TEXT,
            nome_despacho TEXT,
            caminho_despacho TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- LÓGICA DE EXTRAÇÃO E VALIDAÇÃO ---
def extrair_e_validar_merito(texto_completo, nome_arquivo):
    texto_maiusculo = texto_completo.upper()
    pontuacao = 0
    
    # Critérios para garantir que o arquivo é uma petição válida
    evidencias = {
        r'\bEXCELENTÍSSIMO\b': 2, r'\bDOUTO JUÍZO\b': 2, 
        r'\bAÇÃO ORDINÁRIA\b': 2, r'\bPEDE DEFERIMENTO\b': 2
    }

    for padrao, pontos in evidencias.items():
        if re.search(padrao, texto_maiusculo):
            pontuacao += pontos

    inicio_fatos = texto_maiusculo.find("DOS FATOS")
    if inicio_fatos != -1: pontuacao += 3

    if pontuacao < 4:
        raise ValueError(f"O arquivo {nome_arquivo} não possui estrutura jurídica reconhecida.")

    # Isola o mérito (Fatos) para evitar similaridade por cabeçalho
    if inicio_fatos != -1:
        fim_fatos = texto_maiusculo.find("DO DIREITO", inicio_fatos)
        if fim_fatos == -1:
            fim_fatos = texto_maiusculo.find("DOS PEDIDOS", inicio_fatos)
        return texto_completo[inicio_fatos:fim_fatos].strip()
    
    return texto_completo[:4000].strip()

# --- ENDPOINTS ---

@app.post("/armazenar/")
async def armazenar(arquivo_peticao: UploadFile = File(...), arquivo_despacho: UploadFile = File(...)):
    """Salva os PDFs no disco e cria o vínculo semântico no banco."""
    try:
        path_peticao = os.path.join(UPLOAD_DIR, "peticoes", arquivo_peticao.filename)
        path_despacho = os.path.join(UPLOAD_DIR, "despachos", arquivo_despacho.filename)

        # Salva petição
        with open(path_peticao, "wb") as buffer:
            shutil.copyfileobj(arquivo_peticao.file, buffer)
        
        # Salva despacho
        with open(path_despacho, "wb") as buffer:
            shutil.copyfileobj(arquivo_despacho.file, buffer)

        # Processa texto para o modelo NLP
        leitor = PdfReader(path_peticao)
        texto = "".join([p.extract_text() or "" for p in leitor.pages])
        merito = extrair_e_validar_merito(texto, arquivo_peticao.filename)
        vetor = model.encode(merito, convert_to_tensor=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO peticoes (nome_peticao, merito, vetor, nome_despacho, caminho_despacho) 
            VALUES (?, ?, ?, ?, ?)
        ''', (arquivo_peticao.filename, merito, json.dumps(vetor.tolist()), arquivo_despacho.filename, path_despacho))
        conn.commit()
        conn.close()

        return {"status": "sucesso", "despacho_vinculado": arquivo_despacho.filename}
    except Exception as e:
        return {"erro": str(e)}

@app.post("/comparar/")
async def recomendar(arquivo_novo: UploadFile = File(...)):
    """Busca a petição mais similar e retorna o nome/ID do despacho correspondente."""
    try:
        conteudo = await arquivo_novo.read()
        leitor = PdfReader(io.BytesIO(conteudo))
        texto = "".join([p.extract_text() or "" for p in leitor.pages])
        merito_novo = extrair_e_validar_merito(texto, arquivo_novo.filename)
        emb_novo = model.encode(merito_novo, convert_to_tensor=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, nome_peticao, vetor, nome_despacho FROM peticoes")
        registros = cursor.fetchall()
        conn.close()

        if not registros:
            return {"aviso": "O banco de dados está vazio."}

        vetores_base = torch.tensor([json.loads(r[2]) for r in registros])
        similaridades = util.cos_sim(emb_novo, vetores_base)[0]
        score, idx = torch.max(similaridades, dim=0)
        
        percentual = score.item() * 100
        resultado = registros[idx.item()]

        return {
            "similaridade": f"{percentual:.2f}%",
            "peticao_base": resultado[1],
            "despacho_sugerido": resultado[3],
            "download_id": resultado[0] if percentual >= 65 else None
        }
    except Exception as e:
        return {"erro": str(e)}

# @app.get("/baixar-despacho/{id}")
# async def baixar(id: int):
#     """Permite o download do arquivo físico do despacho pelo ID."""
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT nome_despacho, caminho_despacho FROM peticoes WHERE id = ?", (id,))
#     res = cursor.fetchone()
#     conn.close()

#     if not res or not os.path.exists(res[1]):
#         raise HTTPException(status_code=404, detail="Arquivo não encontrado no servidor.")

#     return FileResponse(path=res[1], filename=res[0], media_type='application/pdf')

@app.get("/")
async def raiz():    
    return {"mensagem": "API funcionando!"}

@app.get("/peticoes/")
async def listar():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nome_peticao, nome_despacho FROM peticoes")
    registros = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "peticao": r[1], "despacho": r[2]} for r in registros]

@app.delete("/peticoes/{id}")
async def deletar(id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT nome_peticao, caminho_despacho FROM peticoes WHERE id = ?", (id,))
    res = cursor.fetchone()
    
    if res:
        # Apaga os arquivos físicos para não entulhar o servidor
        p_peticao = os.path.join(UPLOAD_DIR, "peticoes", res[0])
        if os.path.exists(p_peticao): os.remove(p_peticao)
        if os.path.exists(res[1]): os.remove(res[1])
        
        cursor.execute("DELETE FROM peticoes WHERE id = ?", (id,))
        conn.commit()
    
    conn.close()
    return {"status": "sucesso", "mensagem": f"Registro {id} removido."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)