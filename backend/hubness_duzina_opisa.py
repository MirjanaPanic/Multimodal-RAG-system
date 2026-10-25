"""Korelacija duzine opisa sa hubness-om i Hit@1, sirov vs centriran; treba Qdrant"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
PREFIKS_UPITA = "Represent this sentence for searching relevant passages: "
TEZINA_DOKUMENTA = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

klijent = QdrantClient("localhost", port=6333)

with open(KB_DIR / "tekst_upiti.json", encoding="utf-8") as f:
    svi_upiti = json.load(f)
with open(KB_DIR / "knowledge_base.json", encoding="utf-8") as f:
    kb = json.load(f)

vrste = sorted(s for s in svi_upiti if s != "_meta")
prave = [s for s in vrste for _ in svi_upiti[s]["val"]]
upiti = [PREFIKS_UPITA + st["upit"] for s in vrste for st in svi_upiti[s]["val"]]
duzina = np.array([len(kb[s]["tekst"].split()) for s in vrste])
srednji_upita = np.load(KB_DIR / "bge_srednji_vektori.npz")["srednji_vektor_val_upita"]


def centriraj(v, sredina):
    v = v - sredina
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def skorovi(vektor, koristi, nivo):
    odgovor = klijent.query_points_groups(
        collection_name="plant_text", query=vektor.tolist(), using=koristi,
        query_filter=Filter(must=[FieldCondition(key="nivo", match=MatchValue(value=nivo))]),
        group_by="specie_id", limit=len(vrste), group_size=1,
    )
    return {g.id: g.hits[0].score for g in odgovor.groups}


def pobednik(vektor, koristi):   # kao u server.py
    p = skorovi(vektor, koristi, "prozor")
    d = skorovi(vektor, koristi, "dokument")
    s = {sid: (1 - TEZINA_DOKUMENTA) * p[sid] + TEZINA_DOKUMENTA * d[sid] for sid in p}
    return max(s, key=s.get)


def izmeri(model, koristi, centrirano):
    vektori = model.encode(upiti, batch_size=16, normalize_embeddings=True)
    if centrirano:
        vektori = centriraj(vektori, srednji_upita)
    n1 = {s: 0 for s in vrste}
    pogodaka = {s: 0 for s in vrste}
    for v, prava in zip(vektori, prave):
        pob = pobednik(v, koristi)
        n1[pob] += 1
        pogodaka[prava] += int(pob == prava)
    broj_upita = {s: prave.count(s) for s in vrste}
    N1 = np.array([n1[s] for s in vrste])
    hit1 = np.array([pogodaka[s] / broj_upita[s] for s in vrste])
    return N1, hit1


def izvestaj(naziv, N1, hit1):
    print(f"\n=== {naziv} ===")
    r_n, p_n = spearmanr(duzina, N1)
    r_h, p_h = spearmanr(duzina, hit1)
    print(f"Spearman duzina ~ N1  : rho = {r_n:+.3f}, p = {p_n:.3g}")
    print(f"Spearman duzina ~ Hit@1: rho = {r_h:+.3f}, p = {p_h:.3g}")
    granice = np.quantile(duzina, [0.25, 0.5, 0.75])
    kvartil = np.digitize(duzina, granice)   # 0 = najkraci, 3 = najduzi
    print(f"granice kvartila (reci): {granice.astype(int).tolist()}")
    print(f"{'kvartil':<10}{'vrsta':<8}{'prosek reci':<14}{'prosek N1':<12}{'anti (N1=0)':<14}{'Hit@1':<8}")
    for q, ime in enumerate(["Q1 kratki", "Q2", "Q3", "Q4 dugi"]):
        m = kvartil == q
        print(f"{ime:<10}{m.sum():<8}{duzina[m].mean():<14.0f}{N1[m].mean():<12.2f}"
              f"{(N1[m] == 0).mean():<14.0%}{hit1[m].mean():<8.1%}")


model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
izvestaj("bge-large, sirov", *izmeri(model, "sirov", False))
izvestaj("bge-large, centriran (trenutno)", *izmeri(model, "centriran", True))
