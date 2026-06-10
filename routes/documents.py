import sqlite3
import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from config import DB_PATH
from file_handler import salvar_arquivo, extrair_texto_pdf, gerar_vetor_embeddings
from npl_processor import extrair_e_validar_merito

router = APIRouter()


@router.post("/armazenar")
async def armazenar(
    titulo_peticao: str = None,
    titulo_despacho: str = None,
    arquivo_peticao: UploadFile = File(...),
    arquivo_despacho: UploadFile = File(...)
):
    """Salva os PDFs no disco com títulos e cria o vínculo semântico no banco."""
    try:
        path_peticao = salvar_arquivo(arquivo_peticao, "peticoes")
        path_despacho = salvar_arquivo(arquivo_despacho, "despachos")

        # Processa texto para o modelo NLP
        texto = extrair_texto_pdf(path_peticao)
        merito = extrair_e_validar_merito(texto, arquivo_peticao.filename)
        vetor = gerar_vetor_embeddings(merito)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO peticoes (titulo_peticao, nome_peticao, merito, vetor, titulo_despacho, nome_despacho, caminho_peticao, caminho_despacho) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            titulo_peticao or arquivo_peticao.filename,
            arquivo_peticao.filename,
            merito,
            vetor,
            titulo_despacho or arquivo_despacho.filename,
            arquivo_despacho.filename,
            path_peticao,
            path_despacho
        ))
        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()

        return {"status": "sucesso", "id": inserted_id, "despacho_vinculado": arquivo_despacho.filename}
    except Exception as e:
        return {"erro": str(e)}


@router.delete("/peticoes/{id}")
async def deletar(id: int):
    """Deleta um registro de petição e seus arquivos associados."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT nome_peticao, caminho_peticao, caminho_despacho FROM peticoes WHERE id = ?", (id,))
    res = cursor.fetchone()
    
    if res:
        # Apaga os arquivos físicos para não entulhar o servidor
        if os.path.exists(res[1]): 
            os.remove(res[1])
        if os.path.exists(res[2]): 
            os.remove(res[2])
        
        cursor.execute("DELETE FROM peticoes WHERE id = ?", (id,))
        conn.commit()
    
    conn.close()
    return {"status": "sucesso", "mensagem": f"Registro {id} removido."}
