from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Any

import feedparser
import requests

from live_store import connect, upsert_item, get_state, set_state


def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def normalize_entry(entry: Dict[str, Any], source: str) -> Dict[str, Any]:
    uid = entry.get("id") or entry.get("link") or entry.get("title", "")
    published = entry.get("published") or entry.get("updated")
    title = entry.get("title", "") or ""
    summary = entry.get("summary", "") or entry.get("description", "") or ""
    link = entry.get("link")

    return {
        "uid": f"{source}::{uid}",
        "source": source,
        "published": published,
        "title": title.strip(),
        "text": (title + ". " + summary).strip(),
        "url": link,
        "raw_json": json.dumps(entry, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser(description="Ingesta RSS (gratis) robusta: requests + cache + dedupe")
    ap.add_argument("--name", required=True, help="Nombre corto (ej: mit_ai_news)")
    ap.add_argument("--url", required=True, help="URL RSS/Atom")
    ap.add_argument("--sleep", type=float, default=0.3, help="Pausa entre requests")
    args = ap.parse_args()

    con = connect()
    state = get_state(con, args.name)
    log("INFO", f"Fuente={args.name} | etag={state['etag']} | last_modified={state['last_modified']}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LiveTrendsBot/1.0"
    }
    # Si el servidor soporta If-None-Match / If-Modified-Since, los mandamos.
    if state["etag"]:
        headers["If-None-Match"] = state["etag"]
    if state["last_modified"]:
        headers["If-Modified-Since"] = state["last_modified"]

    try:
        r = requests.get(args.url, headers=headers, timeout=25)
        if r.status_code == 304:
            log("INFO", "Sin cambios (304).")
            return
        r.raise_for_status()
        content = r.content
    except Exception as e:
        log("ERROR", f"No se pudo descargar RSS: {e}")
        raise SystemExit(1)

    # Guardar cache si el server lo devuelve
    etag_new = r.headers.get("ETag")
    last_modified_new = r.headers.get("Last-Modified")
    set_state(con, args.name, etag_new, last_modified_new)

    feed = feedparser.parse(content)
    if getattr(feed, "bozo", False):
        log("WARN", f"RSS bozo (parse imperfecto): {getattr(feed, 'bozo_exception', None)}")

    entries = getattr(feed, "entries", []) or []
    new_count = 0

    for e in entries:
        item = normalize_entry(e, args.name)
        if not item["uid"] or item["uid"].endswith("::"):
            continue

        try:
            if upsert_item(con, item):
                new_count += 1
        except Exception as ex:
            log("WARN", f"No se pudo insertar item RSS: {ex}")

    time.sleep(args.sleep)
    log("INFO", f"Items procesados={len(entries)} | nuevos={new_count}")


if __name__ == "__main__":
    main()
