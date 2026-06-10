import sqlite3
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from config import DB_PATH
from file_handler import obter_mime_type

router = APIRouter()


@router.get("/peticoes/")
async def listar():
    """Lista todas as petições cadastradas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, titulo_peticao, titulo_despacho FROM peticoes")
    registros = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "titulo_peticao": r[1], "titulo_despacho": r[2]} for r in registros]


@router.get("/acessar/{doc_id}/{tipo}")
async def acessar_documento(doc_id: int, tipo: str):
    """Visualiza PDF no navegador ou faz download de outros formatos."""
    try:
        if tipo not in ["peticao", "despacho"]:
            raise HTTPException(status_code=400, detail="Tipo deve ser 'peticao' ou 'despacho'")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if tipo == "peticao":
            cursor.execute("SELECT nome_peticao, caminho_peticao FROM peticoes WHERE id = ?", (doc_id,))
        else:
            cursor.execute("SELECT nome_despacho, caminho_despacho FROM peticoes WHERE id = ?", (doc_id,))
        
        res = cursor.fetchone()
        conn.close()

        if not res or not os.path.exists(res[1]):
            raise HTTPException(status_code=404, detail="Arquivo não encontrado no servidor.")

        nome_arquivo = res[0]
        caminho_arquivo = res[1]
        extensao = os.path.splitext(nome_arquivo)[1].lower()

        media_type = obter_mime_type(extensao)
        
        # PDF abre no navegador, outros fazem download
        if extensao == '.pdf':
            disposition = "inline"
        else:
            disposition = "attachment"

        return FileResponse(
            path=caminho_arquivo,
            media_type=media_type,
            headers={"Content-Disposition": f"{disposition}; filename={nome_arquivo}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        return {"erro": str(e)}
