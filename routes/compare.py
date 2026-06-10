import sqlite3
import json
import torch
from fastapi import APIRouter, File, UploadFile
from sentence_transformers import util
from config import DB_PATH, model
from file_handler import extrair_texto_pdf_bytes, gerar_vetor_embeddings
from npl_processor import extrair_e_validar_merito

router = APIRouter()


@router.post("/comparar")
async def recomendar(arquivo_novo: UploadFile = File(...)):
    """Busca a petição mais similar e retorna títulos e IDs dos documentos."""
    try:
        conteudo = await arquivo_novo.read()
        texto = extrair_texto_pdf_bytes(conteudo)
        merito_novo = extrair_e_validar_merito(texto, arquivo_novo.filename)
        emb_novo = model.encode(merito_novo, convert_to_tensor=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, titulo_peticao, vetor, titulo_despacho FROM peticoes")
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
            "titulo_peticao": resultado[1],
            "titulo_despacho": resultado[3],
            "id": resultado[0],
            "pode_acessar": percentual >= 65
        }
    except Exception as e:
        return {"erro": str(e)}
