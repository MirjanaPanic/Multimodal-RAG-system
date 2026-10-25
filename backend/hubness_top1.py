"""Hubness za k=1 pre i posle centriranja na val upitima -> hubness_top1.png; treba Qdrant i plant_text_old"""
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import skew
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

BACKEND_DIR = Path(__file__).resolve().parent
KB_DIR = BACKEND_DIR / "knowledge_base"
PREFIKS_UPITA = "Represent this sentence for searching relevant passages: "
TEZINA_DOKUMENTA = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

klijent = QdrantClient("localhost", port=6333)

with open(KB_DIR / "tekst_upiti.json", encoding="utf-8") as f:
    svi_upiti = json.load(f)
vrste = sorted(s for s in svi_upiti if s != "_meta")
upiti = [PREFIKS_UPITA + st["upit"] for s in vrste for st in svi_upiti[s]["val"]]
print(f"Vrsta: {len(vrste)}, val upita: {len(upiti)}")

srednji = np.load(KB_DIR / "bge_srednji_vektori.npz")
srednji_upita = srednji["srednji_vektor_val_upita"]


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


def pobednik(kolekcija, vektor, koristi, kombinuj):
    if kombinuj:   # kao u server.py
        p = skorovi(kolekcija, vektor, koristi, "prozor")
        d = skorovi(kolekcija, vektor, koristi, "dokument")
        s = {sid: (1 - TEZINA_DOKUMENTA) * p[sid] + TEZINA_DOKUMENTA * d[sid] for sid in p}
    else:
        s = skorovi(kolekcija, vektor, koristi)
    return max(s, key=s.get)


def broji(model, kolekcija, koristi, kombinuj, centrirano):
    vektori = model.encode(upiti, batch_size=16, normalize_embeddings=True)
    if centrirano:
        vektori = centriraj(vektori, srednji_upita)
    brojac = {s: 0 for s in vrste}
    for v in vektori:
        brojac[pobednik(kolekcija, v, koristi, kombinuj)] += 1
    return np.array([brojac[s] for s in vrste])


postavke = {}
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)
postavke["bge-small, sirov (plant_text_old)"] = broji(model, "plant_text_old", "sirov", False, False)
del model
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
postavke["bge-large, centriran (plant_text)"] = broji(model, "plant_text", "centriran", True, True)

NASLOVI = ["Pre centriranja", "Posle centriranja"]
OPISI = ["bge-small, sirovi vektori", "bge-large, centrirani vektori"]
IDEALNO = len(upiti) / len(vrste)   # ravnomerna raspodela upita
N_VRSTA = len(vrste)
GORNJA = max(N.max() for N in postavke.values())
PLAVA, NARANDZASTA = "#1f77b4", "#ff7f0e"

fig, ose = plt.subplots(len(postavke), 1, figsize=(12, 9), sharex=True, sharey=True)
for osa, naslov, opis, N1 in zip(ose, NASLOVI, OPISI, postavke.values()):
    # hub: N > prosek + 2 std, anti-hub: N = 0
    prag_hub = N1.mean() + 2 * N1.std()
    je_hub = N1 > prag_hub
    je_anti = N1 == 0
    x = np.arange(N_VRSTA)
    osa.bar(x[~je_hub], N1[~je_hub], width=1.0, align="edge", color=PLAVA)
    osa.bar(x[je_hub], N1[je_hub], width=1.0, align="edge", color=NARANDZASTA)
    osa.scatter(x[je_anti] + 0.5, np.zeros(je_anti.sum()), marker="^", s=16, color="gray",
                clip_on=False, zorder=3)
    osa.axhline(IDEALNO, color="red", linestyle="--", linewidth=1.2)
    osa.set_title(f"{naslov} ({opis})\n"
                  f"čvorišta: {int(je_hub.sum())}, anti-čvorišta: {int(je_anti.sum())} "
                  f"({je_anti.mean():.0%} vrsta), skewness = {skew(N1):.2f}", fontsize=12)
    osa.set_ylabel("Broj upita u kojima je vrsta na 1. mestu")
    osa.set_xlabel(f"Biljne vrste (redni broj od 0 do {N_VRSTA}, redosledom iz baze znanja)")
    osa.set_xlim(0, N_VRSTA)
    osa.set_ylim(0, GORNJA * 1.08)
    osa.set_xticks(np.arange(0, N_VRSTA, 50))
    osa.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    osa.tick_params(labelbottom=True)
    osa.grid(axis="y", alpha=0.25)
    osa.set_axisbelow(True)
    osa.spines[["top", "right"]].set_visible(False)
    print(f"{ime if (ime := opis) else ''}: max={N1.max()}, nula={je_anti.sum()}/{N_VRSTA}, "
          f"hubova={je_hub.sum()}, skewness={skew(N1):.2f}")

ruchke = [
    Patch(color=PLAVA, label="obične vrste"),
    Patch(color=NARANDZASTA, label="čvorišta (hubs): vrste sa više upita od praga (prosek + 2 std)"),
    Line2D([], [], color="red", linestyle="--",
           label=f"očekivani broj upita po vrsti ({len(upiti)} upita / {N_VRSTA} vrsta = {IDEALNO:.0f})"),
    Line2D([], [], color="gray", marker="^", linestyle="", markersize=6,
           label="anti-čvorišta (anti-hubs): vrsta nije bila prva ni u jednom upitu"),
]
fig.legend(handles=ruchke, loc="lower center", ncol=2, frameon=False, fontsize=10)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("hubness_top1.png", dpi=150)
print("Sacuvano: hubness_top1.png")
