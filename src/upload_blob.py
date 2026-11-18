"""Publish the model and panel artifacts to Vercel Blob.

Needs BLOB_READ_WRITE_TOKEN in .env (never printed). The token is only sent to
blob.vercel-storage.com.

  python src/upload_blob.py list      what is in the store
  python src/upload_blob.py upload    upload the artifacts and print the env vars to set
"""
import argparse
import hashlib
import os
from pathlib import Path

import dotenv
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://blob.vercel-storage.com"
API_VERSION = "7"
PREFIX = "ephpm/v3"

# env var the API reads -> local file
ARTIFACTS = {
    "MODEL_BLOB_URL": ROOT / "models" / "clustering_panel.onnx",
    "MODEL_META_BLOB_URL": ROOT / "models" / "clustering_panel.json",
    "PANEL_BLOB_URL": ROOT / "api_data" / "panel.json",
    "POVERTY_MODEL_BLOB_URL": ROOT / "models" / "poverty_logreg.onnx",
    "POVERTY_META_BLOB_URL": ROOT / "models" / "poverty_logreg.json",
}
CONTENT_TYPE = {".json": "application/json", ".onnx": "application/octet-stream"}


def headers() -> dict:
    dotenv.load_dotenv(ROOT / ".env")
    token = os.getenv("BLOB_READ_WRITE_TOKEN")
    if not token:
        raise SystemExit("Falta BLOB_READ_WRITE_TOKEN en .env")
    return {"authorization": f"Bearer {token}", "x-api-version": API_VERSION}


def list_blobs() -> list:
    r = requests.get(API, params={"limit": 1000}, headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json()["blobs"]


def upload(path: Path) -> str:
    r = requests.put(
        f"{API}/{PREFIX}/{path.name}",
        data=path.read_bytes(),
        headers={**headers(),
                 "x-content-type": CONTENT_TYPE[path.suffix],
                 "x-add-random-suffix": "0",
                 "x-allow-overwrite": "1"},
        timeout=60,
    )
    if not r.ok:
        raise SystemExit(f"Error {r.status_code} subiendo {path.name}: {r.text[:300]}")
    return r.json()["url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "upload"])
    args = ap.parse_args()

    if args.cmd == "list":
        for b in list_blobs():
            print(f"{b['size']:>9,d}  {b['uploadedAt'][:19]}  {b['pathname']}")
        return

    for var, path in ARTIFACTS.items():
        url = upload(path)
        # Read it back without credentials: this is what the API will do in production.
        remote = requests.get(url, params={"v": hashlib.md5(path.read_bytes()).hexdigest()}, timeout=30)
        same = remote.ok and remote.content == path.read_bytes()
        print(f"{var}={url}   {'verificado' if same else 'NO COINCIDE'}")


if __name__ == "__main__":
    main()
