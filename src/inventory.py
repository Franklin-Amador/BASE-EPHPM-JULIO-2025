"""Inventory of every EPHPM .sav under data/raw: size and candidate columns
for the variables the pipeline needs.

Usage: python src/inventory.py [--out data/processed/inventory.json]
"""
import argparse
import json
import re
from pathlib import Path

import pyreadstat

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# concept -> (regex on column name, regex on column label)
CONCEPTS = {
    "depto": (r"^(depto|dpto|departamento)$", r"departamento"),
    "dominio": (r"^(dominio|domi|ur|area)$", r"dominio|urbano"),
    "hogar_id": (r"^(hogar|numhog|nhogar|id_?hogar|vivienda|numviv)$", r"n[uú]mero de hogar|hogar"),
    "nper": (r"^(nper|orden|n_?linea)$", r"n[uú]mero de (la )?persona|orden"),
    "factor": (r"^(factor|fact|peso|pondera).*", r"factor|ponderad"),
    "edad": (r"^(edad|p0?\d*_?edad)$", r"^edad|edad en a[nñ]os"),
    "totper": (r"^(totper|tot_?per|totpers)$", r"total de personas|personas en el hogar"),
    "anosest": (r"^a[nñ]o?s?est.*", r"a[nñ]os de estudio"),
    "ymophg": (r"^(ymophg|ymop_?hg|ythg|yperhg|ymohg).*", r"ingreso.*hogar"),
    "condact": (r"^(condact|cond_?act).*", r"condici[oó]n de actividad"),
    "pobreza": (r"^pobreza.*", r"pobreza"),
    "nbi": (r"^nbi.*", r"necesidades b[aá]sicas"),
}


def year_of(path: Path) -> int:
    m = re.search(r"(19|20)\d{2}", path.stem)
    return int(m.group()) if m else int(path.parent.name[:4])


def candidates(meta, name_rx, label_rx):
    out = []
    for name, label in zip(meta.column_names, meta.column_labels):
        if re.search(name_rx, name, re.I) or (label and re.search(label_rx, label, re.I)):
            out.append({"name": name, "label": label})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW.parent / "processed" / "inventory.json"))
    args = ap.parse_args()

    report = {}
    for sav in sorted(RAW.rglob("*.sav"), key=year_of):
        year = year_of(sav)
        try:
            _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
        except Exception as exc:  # corrupt or unusual encoding
            report[year] = {"file": sav.name, "error": str(exc)}
            print(f"{year}  ERROR {sav.name}: {exc}")
            continue
        found = {c: candidates(meta, *rx) for c, rx in CONCEPTS.items()}
        report[year] = {
            "file": str(sav.relative_to(RAW)),
            "rows": meta.number_rows,
            "cols": meta.number_columns,
            "candidates": found,
        }
        print(f"\n{year}  {sav.name}  rows={meta.number_rows} cols={meta.number_columns}")
        for concept, cands in found.items():
            names = ", ".join(c["name"] for c in cands[:8]) or "--"
            print(f"  {concept:9s} {names}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
