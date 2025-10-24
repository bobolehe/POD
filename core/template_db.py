import os
import json
import sqlite3
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    mask_meta TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DB_PATH)


def init_db(db_path: Optional[str] = None) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def add_template(name: str, path: str, width: Optional[int] = None, height: Optional[int] = None,
                 mask_meta: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None) -> int:
    conn = get_conn(db_path)
    try:
        meta_json = json.dumps(mask_meta) if mask_meta is not None else None
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO templates(name, path, width, height, mask_meta) VALUES(?,?,?,?,?)",
            (name, path, width, height, meta_json)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_templates(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, path, width, height, mask_meta, created_at FROM templates ORDER BY id DESC")
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r[5]) if r[5] else None
            out.append({
                'id': r[0], 'name': r[1], 'path': r[2], 'width': r[3], 'height': r[4], 'mask_meta': meta, 'created_at': r[6]
            })
        return out
    finally:
        conn.close()


def get_template(tid: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, path, width, height, mask_meta, created_at FROM templates WHERE id=?", (tid,))
        r = cur.fetchone()
        if not r:
            return None
        meta = json.loads(r[5]) if r[5] else None
        return {
            'id': r[0], 'name': r[1], 'path': r[2], 'width': r[3], 'height': r[4], 'mask_meta': meta, 'created_at': r[6]
        }
    finally:
        conn.close()


def delete_template(tid: int, db_path: Optional[str] = None) -> bool:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM templates WHERE id=?", (tid,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_template(tid: int, updates: Dict[str, Any], db_path: Optional[str] = None) -> bool:
    conn = get_conn(db_path)
    try:
        fields = []
        values = []
        for k, v in updates.items():
            if k == 'mask_meta' and v is not None:
                v = json.dumps(v)
            fields.append(f"{k} = ?")
            values.append(v)
        if not fields:
            return False
        values.append(tid)
        sql = f"UPDATE templates SET {', '.join(fields)} WHERE id = ?"
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()