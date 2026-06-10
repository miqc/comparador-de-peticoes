import os
import shutil
import json
from io import BytesIO
from pypdf import PdfReader
from config import model, UPLOAD_DIR


def salvar_arquivo(arquivo, tipo_arquivo):
    """Salva arquivo PDF no diretório apropriado."""
    path = os.path.join(UPLOAD_DIR, tipo_arquivo, arquivo.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(arquivo.file, buffer)
    return path


def extrair_texto_pdf(path_pdf):
    """Extrai texto de um arquivo PDF."""
    leitor = PdfReader(path_pdf)
    texto = "".join([p.extract_text() or "" for p in leitor.pages])
    return texto


def extrair_texto_pdf_bytes(conteudo):
    """Extrai texto de um PDF a partir de bytes."""
    leitor = PdfReader(BytesIO(conteudo))
    texto = "".join([p.extract_text() or "" for p in leitor.pages])
    return texto


def gerar_vetor_embeddings(merito_texto):
    """Gera vetor de embeddings para o texto usando o modelo NLP."""
    vetor = model.encode(merito_texto, convert_to_tensor=True)
    return json.dumps(vetor.tolist())


def obter_mime_type(extensao):
    """Retorna o MIME type baseado na extensão do arquivo."""
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.txt': 'text/plain',
    }
    return mime_types.get(extensao, 'application/octet-stream')


def deletar_arquivo(caminho):
    """Deleta um arquivo se existir."""
    if os.path.exists(caminho):
        os.remove(caminho)
