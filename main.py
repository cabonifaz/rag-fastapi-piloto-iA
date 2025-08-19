# app/main.py

from fastapi import FastAPI
from app.api import chat
from app.infrastructure.db.database import init_db

# Crear la aplicación FastAPI
app = FastAPI(
    title="RAG-FastAPI",
    description="Retrieval-Augmented Generation with FastAPI (Hexagonal Architecture)",
    version="0.1.0"
)

# Iniciar conexión a la base de datos al levantar el servidor
@app.on_event("startup")
def on_startup():
    init_db()

# Incluir routers (endpoints)
app.include_router(chat.router, prefix="/api", tags=["chat"])
