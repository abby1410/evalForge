import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "evalforge.db"

def connect(db_path=None):
    conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=None):
    with connect(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            total_cases INTEGER NOT NULL,
            passed_cases INTEGER NOT NULL,
            average_score REAL NOT NULL,
            average_latency_ms REAL NOT NULL,
            results_json TEXT NOT NULL
        )""")
        conn.commit()

def save_run(run, db_path=None):
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute("""INSERT INTO runs
        (created_at,label,prompt_version,prompt,provider,model,judge_model,dataset_name,total_cases,passed_cases,average_score,average_latency_ms,results_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            run["created_at"], run["label"], run["prompt_version"], run["prompt"],
            run["provider"], run["model"], run["judge_model"], run["dataset_name"],
            run["total_cases"], run["passed_cases"], run["average_score"],
            run["average_latency_ms"], json.dumps(run["results"], ensure_ascii=False)
        ))
        conn.commit()
        return cur.lastrowid

def list_runs(db_path=None):
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

def get_run(run_id, db_path=None):
    for row in list_runs(db_path):
        if row["id"] == run_id:
            row["results"] = json.loads(row["results_json"])
            return row
    return None

def build_run_record(label, prompt_version, prompt, provider, model, judge_model, dataset_name, results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["overall_score"] for r in results) / max(1, total)
    avg_latency = sum(r["latency_ms"] for r in results) / max(1, total)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "prompt_version": prompt_version, "prompt": prompt,
        "provider": provider, "model": model, "judge_model": judge_model,
        "dataset_name": dataset_name, "total_cases": total, "passed_cases": passed,
        "average_score": round(avg_score, 3), "average_latency_ms": round(avg_latency, 1),
        "results": results,
    }
