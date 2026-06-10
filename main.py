from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import CORS_ORIGINS
from database import init_db
from routes import documents_router, compare_router, listing_router

# Inicializa o banco de dados
init_db()

# Criar aplicação FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rotas
app.include_router(documents_router)
app.include_router(compare_router)
app.include_router(listing_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)