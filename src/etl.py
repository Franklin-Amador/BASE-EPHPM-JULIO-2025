"""Multi-year EPHPM ETL.

Reads the person-level .sav files listed in config/variables.yml and writes:
  data/processed/personas.parquet    harmonised person rows, all years
  data/processed/hogares.parquet     one row per household and year
  data/processed/depto_anio.parquet  the 8 indicators per department and year (+ .csv)

All department indicators are weighted with the expansion factor.

Usage: python src/etl.py [--years 2023 2024 2025]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "variables.yml"

FEATURES = [
    "ymophg_mean", "ymophg_median", "anosest_mean", "edad_mean",
    "totper_mean", "tasa_ocupacion", "tasa_pobreza", "tasa_nbi",
]


VALID_RANGE = {"edad": (0, 110), "anosest": (0, 25), "totper": (1, 40)}
WORKING_AGE = 15  # same base for the occupation rate in every year


def load_year(year: int, spec: dict, cfg: dict) -> pd.DataFrame:
    """Return harmonised person rows for one survey year."""
    path = RAW / spec["file"]
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    lower = {c.lower(): c for c in meta.column_names}

    cols = {}
    for concept, names in cfg["aliases"].items():
        hit = next((lower[n.lower()] for n in names if n.lower() in lower), None)
        if hit:
            cols[concept] = hit
    df, _ = pyreadstat.read_sav(str(path), usecols=list(cols.values()))
    df = df.rename(columns={v: k for k, v in cols.items()})
    for concept in cfg["aliases"]:
        if concept not in df:
            df[concept] = np.nan

    if spec.get("depto_from_cor_pre"):
        df["depto"] = df["cor_pre"] // 10000
    if spec.get("swap_depto_9_11"):
        df["depto"] = df["depto"].replace({9: 11, 11: 9})
    df.loc[~df["depto"].isin(cfg["depto_names"].keys()), "depto"] = np.nan
    # Some files store "no sabe / no responde" as 99, 999 or 9999.
    for col, (lo, hi) in VALID_RANGE.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan

    df["anio"] = year
    hid = df["hogar_id"]
    hid = hid.astype("Int64").astype(str) if pd.api.types.is_numeric_dtype(hid) else hid.astype(str).str.strip()
    df["hogar_key"] = hid + "_" + df["num_hog"].astype("Int64").astype(str)
    df["ymophg_derivado"] = "ymophg" not in cols
    df["tiene_yhreme"] = "yhreme" in cols
    return df.drop(columns=["hogar_id", "num_hog", "cor_pre"])


def educ_level(anosest: pd.Series) -> pd.Series:
    """0 = under 6 years, 1 = 6 to 12, 2 = 13 or more.

    The 2023 and 2024 files code upper-secondary (10-12 years) as 9, so years of schooling
    are not comparable in that range; these three bands are unaffected by it.
    """
    return pd.cut(anosest, [-1, 5, 12, 30], labels=False).astype(float)


def to_households(p: pd.DataFrame) -> pd.DataFrame:
    """Collapse person rows to one row per household."""
    g = p.groupby(["anio", "hogar_key"], sort=False)
    h = g.agg(
        depto=("depto", "first"),
        factor=("factor", "first"),
        totper=("totper", "first"),
        n_personas=("edad", "size"),
        ymophg=("ymophg", "max"),
        ymop_sum=("ymop", lambda s: s.sum(min_count=1)),
        pobreza=("pobreza", "max"),
        nbi=("nbi", "max"),
        anosest_max=("anosest", "max"),
        edad_mean=("edad", "mean"),
        n_ocupados=("condact", lambda s: (s == 1).sum()),
        n_dependientes=("edad", lambda s: ((s < 15) | (s >= 65)).sum()),
        n_menores=("edad", lambda s: (s < 15).sum()),
        n_mayores=("edad", lambda s: (s >= 65).sum()),
        ur=("ur", "first"),
        yperhg=("yperhg", "max"),
        yhreme=("yhreme", "max"),
        tiene_yhreme=("tiene_yhreme", "first"),
        ymophg_derivado=("ymophg_derivado", "first"),
    ).reset_index()
    jefes = p[p["rela_j"] == 1].drop_duplicates(["anio", "hogar_key"])
    jefes = jefes.assign(jefa_mujer=(jefes["sexo"] == 2).astype(float).where(jefes["sexo"].notna()))
    jefes = jefes.assign(
        jefe_ocupado=(jefes["condact"] == 1).astype(float),
        jefe_agricultura=(jefes["ramaop"] == 1).astype(float),
        jefe_educ=educ_level(jefes["anosest"]),
    )
    jefes = jefes.rename(columns={"edad": "jefe_edad", "anosest": "jefe_anosest"})
    h = h.merge(jefes[["anio", "hogar_key", "jefa_mujer", "jefe_edad", "jefe_anosest",
                       "jefe_ocupado", "jefe_agricultura", "jefe_educ"]],
                on=["anio", "hogar_key"], how="left")
    h["max_educ"] = educ_level(h["anosest_max"])
    # Where the file has the remittance column, a blank means the household receives none.
    h["recibe_remesas"] = (h["yhreme"].fillna(0) > 0).astype(float).where(h["tiene_yhreme"])
    # Files without YMOPHG (2021, 2022): household income = sum of members' YMOP.
    # On 2023, where both exist, sum(YMOP) equals YMOPHG for 100% of households.
    h["ymophg"] = h["ymophg"].where(~h["ymophg_derivado"], h["ymop_sum"])
    h.loc[h["ymophg"] <= 0, "ymophg"] = np.nan
    h["pobre"] = h["pobreza"].map({1: 1.0, 2: 1.0, 3: 0.0})
    h["pobre_extremo"] = h["pobreza"].map({1: 1.0, 2: 0.0, 3: 0.0})
    h["con_nbi"] = (h["nbi"] >= 1).astype(float).where(h["nbi"].notna())
    return h.drop(columns=["ymop_sum", "yhreme", "tiene_yhreme"])


def wmean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna()
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def wmedian(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna()
    if not ok.any():
        return np.nan
    order = np.argsort(x[ok].to_numpy())
    xs, ws = x[ok].to_numpy()[order], w[ok].to_numpy()[order]
    return float(xs[np.searchsorted(np.cumsum(ws), ws.sum() / 2)])


def aggregate(p: pd.DataFrame, h: pd.DataFrame, by: list) -> pd.DataFrame:
    """Weighted indicators for each group in `by` (e.g. year, or year x department)."""
    persons = dict(tuple(p.groupby(by)))
    rows = []
    for key, hg in h.groupby(by):
        pg = persons[key]
        tg = pg[pg["edad"] >= WORKING_AGE]  # base of the occupation rate
        key = key if isinstance(key, tuple) else (key,)
        ocup = tg[tg["condact"] == 1]
        activos = tg[tg["condact"].isin([1, 2])]
        adultos = pg[pg["edad"] >= 25]
        dep = (pg["edad"] < 15) | (pg["edad"] >= 65)
        rows.append({
            **dict(zip(by, key)),
            "ymophg_mean": wmean(hg["ymophg"], hg["factor"]),
            "ymophg_median": wmedian(hg["ymophg"], hg["factor"]),
            "anosest_mean": wmean(pg["anosest"], pg["factor"]),
            "edad_mean": wmean(pg["edad"], pg["factor"]),
            "totper_mean": wmean(hg["totper"], hg["factor"]),
            "tasa_ocupacion": wmean((tg["condact"] == 1).astype(float), tg["factor"]),
            "tasa_pobreza": wmean(hg["pobre"], hg["factor"]),
            "tasa_pobreza_extrema": wmean(hg["pobre_extremo"], hg["factor"]),
            "tasa_nbi": wmean(hg["con_nbi"], hg["factor"]),
            "tasa_desempleo": wmean((activos["condact"] == 2).astype(float), activos["factor"]),
            "pct_rural": wmean((pg["ur"] == 2).astype(float).where(pg["ur"].notna()), pg["factor"]),
            "pct_jefatura_femenina": wmean(hg["jefa_mujer"], hg["factor"]),
            "tasa_dependencia": pg.loc[dep, "factor"].sum() / pg.loc[~dep & pg["edad"].notna(), "factor"].sum(),
            "pct_media_completa": wmean((adultos["anosest"] >= 12).astype(float).where(adultos["anosest"].notna()), adultos["factor"]),
            "pct_agricultura": wmean((ocup["ramaop"] == 1).astype(float).where(ocup["ramaop"].notna()), ocup["factor"]),
            "pct_cuenta_propia": wmean((ocup["categop"] == 4).astype(float).where(ocup["categop"].notna()), ocup["factor"]),
            "yperhg_mean": wmean(hg["yperhg"], hg["factor"]),
            "pct_recibe_remesas": wmean(hg["recibe_remesas"], hg["factor"]),
            "n_hogares": len(hg),
            "n_personas": len(pg),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    years = {y: s for y, s in cfg["years"].items() if not args.years or y in args.years}

    personas = pd.concat([load_year(y, s, cfg) for y, s in years.items()], ignore_index=True)
    hogares = to_households(personas)

    nacional = aggregate(personas, hogares, ["anio"])
    with_depto = hogares["depto"].notna()
    depto = aggregate(personas[personas["depto"].notna()], hogares[with_depto], ["anio", "depto"])
    depto["depto"] = depto["depto"].astype(int)
    # Income in current lempiras is not comparable across years; this ratio is.
    nat_median = depto["anio"].map(nacional.set_index("anio")["ymophg_median"])
    depto["ymophg_median_rel"] = depto["ymophg_median"] / nat_median
    depto.insert(2, "depto_nombre", depto["depto"].map(cfg["depto_names"]))
    hogares["depto_nombre"] = hogares["depto"].map(cfg["depto_names"])

    OUT.mkdir(parents=True, exist_ok=True)
    personas.to_parquet(OUT / "personas.parquet", index=False)
    hogares.to_parquet(OUT / "hogares.parquet", index=False)
    depto.to_parquet(OUT / "depto_anio.parquet", index=False)
    depto.to_csv(OUT / "depto_anio.csv", index=False, encoding="utf-8")
    nacional.to_csv(OUT / "nacional_anio.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Nacional por año (ponderado):")
    print(nacional.round(3).T.to_string(header=False))
    print("\nDepartamentos con dato por año y variable:")
    extra = [c for c in depto.columns if c.startswith(("pct_", "tasa_d", "yperhg", "ymophg_median_rel"))]
    print(depto.groupby("anio")[FEATURES + extra].count().T.to_string())
    print(f"\npersonas={len(personas):,} hogares={len(hogares):,} filas depto×año={len(depto)}")


if __name__ == "__main__":
    main()
