import os
from sentence_transformers import SentenceTransformer

# Configuração de banco de dados
DB_PATH = "peticoes_juridicas.db"

# Configuração de diretórios
UPLOAD_DIR = "arquivos_cartorio"
os.makedirs(f"{UPLOAD_DIR}/peticoes", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/despachos", exist_ok=True)

# Configuração do modelo NLP
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Configuração de CORS
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
