"""Hit@1/3/5/10 po tezini celog opisa -> rezultati/hitrate_tezina_bge_large.csv"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BACKEND_DIR = Path(__file__).resolve().parent
KB_DIR = BACKEND_DIR / "knowledge_base"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
PREFIKS_UPITA = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 16
TEZINA_DOKUMENTA = 0.3
KS = [1, 3, 5, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(KB_DIR / "knowledge_base.json", encoding="utf-8") as f:
    knowledge_base = json.load(f)
with open(KB_DIR / "tekst_upiti.json", encoding="utf-8") as f:
    svi_upiti = json.load(f)

specie_ids = sorted(knowledge_base)
BROJ_VRSTA = len(specie_ids)
INDEKS = {sid: i for i, sid in enumerate(specie_ids)}
val_upiti, tacne = [], []
for sid in specie_ids:
    for st in svi_upiti[sid]["val"]:
        val_upiti.append(PREFIKS_UPITA + st["upit"])
        tacne.append(INDEKS[sid])
tacne = np.array(tacne)

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
tokenizer = model.tokenizer


def broj_tokena(tekst):
    return len(tokenizer.encode(tekst, add_special_tokens=False))


# segmentacija iz nb 07
def ocisti_naslove(tekst):
    tekst = re.sub(r"\s*^=+[^\n]*=+\s*\Z", "", tekst, flags=re.MULTILINE)
    tekst = re.sub(r"^=+\s*(.+?)\s*=+[ \t]*\n\s*", r"\1: ", tekst, flags=re.MULTILINE)
    return tekst.strip()


KRAJ_RECENICE = r"(?<=[.!?])\s+(?=[A-Z])"
SEPARATORI = ["\n\n\n", "\n\n", "\n", KRAJ_RECENICE]


def podeli(tekst, separator):
    delovi = re.split(KRAJ_RECENICE, tekst) if separator == KRAJ_RECENICE else tekst.split(separator)
    return [d.strip() for d in delovi if d.strip() != ""]


def podeli_na_chunkove(tekst, max_tokena=500, nivo=0):
    if broj_tokena(tekst) <= max_tokena or nivo == len(SEPARATORI):
        return [tekst]
    separator = SEPARATORI[nivo]
    celine = podeli(tekst, separator)
    if len(celine) == 1:
        return podeli_na_chunkove(tekst, max_tokena, nivo + 1)
    delovi = []
    for celina in celine:
        if broj_tokena(celina) <= max_tokena:
            delovi.append(celina)
        else:
            delovi += podeli_na_chunkove(celina, max_tokena, nivo + 1)
    spojnica = " " if separator == KRAJ_RECENICE else separator
    return spoji_u_chunkove(delovi, spojnica, max_tokena)


def spoji_u_chunkove(celine, spojnica, max_tokena):
    ukupno = sum(broj_tokena(c) for c in celine)
    cilj = ukupno / math.ceil(ukupno / max_tokena)
    chunkovi, tekuci, tekuci_tokeni = [], [], 0
    for celina in celine:
        n = broj_tokena(celina)
        if len(tekuci) > 0:
            novo = tekuci_tokeni + n
            if novo > max_tokena or abs(novo - cilj) > abs(tekuci_tokeni - cilj):
                chunkovi.append(spojnica.join(tekuci))
                tekuci, tekuci_tokeni = [], 0
        tekuci.append(celina)
        tekuci_tokeni += n
    chunkovi.append(spojnica.join(tekuci))
    if len(chunkovi) > 1 and broj_tokena(chunkovi[-2]) + broj_tokena(chunkovi[-1]) <= max_tokena:
        poslednji = chunkovi.pop()
        chunkovi[-1] = chunkovi[-1] + spojnica + poslednji
    return chunkovi


def recenice(tekst):
    lista = []
    for red in podeli(tekst, "\n"):
        lista += podeli(red, KRAJ_RECENICE)
    return lista


def prozori(tekst, sirina=2, korak=1):
    r = recenice(tekst)
    if len(r) <= sirina:
        return [" ".join(r)]
    poceci = list(range(0, len(r) - sirina + 1, korak))
    if poceci[-1] != len(r) - sirina:
        poceci.append(len(r) - sirina)
    return [" ".join(r[i:i + sirina]) for i in poceci]


NACINI = {
    "ceo opis (chunkovi do 500 tokena)": lambda t: podeli_na_chunkove(t, 500),
    "jedna rečenica": recenice,
    "prozor 2 rečenice, bez preklapanja": lambda t: prozori(t, 2, 2),
    "prozor 2 rečenice, sa preklapanjem": lambda t: prozori(t, 2, 1),
    "prozor 3 rečenice, sa preklapanjem": lambda t: prozori(t, 3, 1),
}
ociscen = {sid: ocisti_naslove(knowledge_base[sid]["tekst"]) for sid in specie_ids}


def centriraj(v, mu):
    v = v - mu
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def kodiraj(tekstovi):
    return model.encode(tekstovi, batch_size=BATCH_SIZE, normalize_embeddings=True)


def napravi(nacin):
    delovi, vrste = [], []
    for j, sid in enumerate(specie_ids):
        d = NACINI[nacin](ociscen[sid])
        delovi += d
        vrste += [j] * len(d)
    return kodiraj(delovi), np.array(vrste)


def skor_vrste(sim, vrste):
    return np.column_stack([sim[:, vrste == j].max(axis=1) for j in range(BROJ_VRSTA)])


def hit_rate(S):
    # 1 + broj vrsta sa strogo boljim skorom
    rang = 1 + (S > S[np.arange(len(S)), tacne][:, None]).sum(axis=1)
    return {f"Hit@{k}": round((rang <= k).mean() * 100, 1) for k in KS}



TEZINE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

upiti_sirovi = kodiraj(val_upiti)
upiti_centrirani = centriraj(upiti_sirovi, upiti_sirovi.mean(axis=0))

prozori_vek, prozori_vrste = napravi("prozor 2 rečenice, sa preklapanjem")
S_prozor = skor_vrste(upiti_centrirani @ centriraj(prozori_vek, prozori_vek.mean(axis=0)).T, prozori_vrste)
ceo_vek, ceo_vrste = napravi("ceo opis (chunkovi do 500 tokena)")
S_ceo = skor_vrste(upiti_centrirani @ centriraj(ceo_vek, ceo_vek.mean(axis=0)).T, ceo_vrste)

redovi = []
for w in TEZINE:
    redovi.append({"w (ceo opis)": w, "1 - w (prozor)": round(1 - w, 1),
                   **hit_rate((1 - w) * S_prozor + w * S_ceo)})
tabela = pd.DataFrame(redovi)
pd.set_option("display.width", 200)
print(tabela.to_string(index=False))
tabela.to_csv(BACKEND_DIR / "rezultati" / "hitrate_tezina_bge_large.csv", index=False, encoding="utf-8-sig")
