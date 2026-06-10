import sqlite3
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS peticoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo_peticao TEXT,
            nome_peticao TEXT,
            merito TEXT,
            vetor TEXT,
            titulo_despacho TEXT,
            nome_despacho TEXT,
            caminho_peticao TEXT,
            caminho_despacho TEXT
        )
    ''')
    conn.commit()
    conn.close()
