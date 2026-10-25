"""Hit@1/3/5/10 tekstualne pretrage na val upitima za tri postavke; treba Qdrant i plant_text_old"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
PREFIKS_UPITA = "Represent this sentence for searching relevant passages: "
TEZINA_DOKUMENTA = 0.3
KS = [1, 3, 5, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

klijent = QdrantClient("localhost", port=6333)

with open(KB_DIR / "tekst_upiti.json", encoding="utf-8") as f:
    svi_upiti = json.load(f)
vrste = sorted(s for s in svi_upiti if s != "_meta")
prave = [s for s in vrste for _ in svi_upiti[s]["val"]]
upiti = [PREFIKS_UPITA + st["upit"] for s in vrste for st in svi_upiti[s]["val"]]
srednji_upita = np.load(KB_DIR / "bge_srednji_vektori.npz")["srednji_vektor_val_upita"]


def centriraj(v, sredina):
    v = v - sredina
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def skorovi(kolekcija, vektor, koristi, nivo=None):
    odgovor = klijent.query_points_groups(
        collection_name=kolekcija, query=vektor.tolist(), using=koristi,
        query_filter=(Filter(must=[FieldCondition(key="nivo", match=MatchValue(value=nivo))])
                      if nivo else None),
        group_by="specie_id", limit=len(vrste), group_size=1,
    )
    return {g.id: g.hits[0].score for g in odgovor.groups}


def rang_vrsta(kolekcija, vektor, koristi, kombinuj):
    if kombinuj:
        p = skorovi(kolekcija, vektor, koristi, "prozor")
        d = skorovi(kolekcija, vektor, koristi, "dokument")
        s = {sid: (1 - TEZINA_DOKUMENTA) * p[sid] + TEZINA_DOKUMENTA * d[sid] for sid in p}
    else:
        s = skorovi(kolekcija, vektor, koristi)
    return sorted(s, key=s.get, reverse=True)


def hit_rate(model, kolekcija, koristi, kombinuj, centrirano):
    vektori = model.encode(upiti, batch_size=16, normalize_embeddings=True)
    if centrirano:
        vektori = centriraj(vektori, srednji_upita)
    pozicije = []   # mesto prave vrste
    for v, prava in zip(vektori, prave):
        pozicije.append(rang_vrsta(kolekcija, v, koristi, kombinuj).index(prava) + 1)
    pozicije = np.array(pozicije)
    return {k: (pozicije <= k).mean() for k in KS}


rezultati = {}
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)
rezultati["1. bge-small, sirov, chunkovi do 500 tokena (plant_text_old)"] = hit_rate(model, "plant_text_old", "sirov", False, False)
del model
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
rezultati["2. bge-large, sirov, prozor + dokument"] = hit_rate(model, "plant_text", "sirov", True, False)
rezultati["3. bge-large, centriran, prozor + dokument (trenutno)"] = hit_rate(model, "plant_text", "centriran", True, True)

print(f"\nVal upita: {len(upiti)}, vrsta: {len(vrste)} (slucajan pogodak Hit@1 = {1 / len(vrste):.2%})\n")
print(f"{'postavka':<62}" + "".join(f"Hit@{k:<5}" for k in KS))
for ime, r in rezultati.items():
    print(f"{ime:<62}" + "".join(f"{r[k]:<10.1%}" for k in KS))
