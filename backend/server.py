import io
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

BACKEND_DIR = Path(__file__).parent
SLIKE_DIR = BACKEND_DIR / "dataset" / "train_ready"
KB_DIR = BACKEND_DIR / "knowledge_base"

QDRANT_URL = "http://localhost:6333"
KOLEKCIJA_SLIKE = "plant_images"
KOLEKCIJA_TEKST = "plant_text"

# slikovna grana
MODEL_SLIKE = "hf-hub:imageomics/bioclip"
K = 200
PRAG_ABSTAIN_SLIKA = 0.59
TAU_SLIKA = 0.06

# tekstualna grana
MODEL_TEKSTA = "BAAI/bge-large-en-v1.5"
PREFIKS_UPITA = "Represent this sentence for searching relevant passages: "
TEZINA_DOKUMENTA = 0.3
PRAG_ABSTAIN_TEKST = 0.6257
TAU_TEKST = 0.1162

# fuzija
DUZINA_LISTE = 3
K_RRF = 10
W_SLIKA = 0.6

# ograničenja unosa
MAX_SLIKA = 5
MAX_MB = 10
MIN_STRANICA = 224
DOZVOLJENI_FORMATI = {"JPEG", "PNG", "JPG"}
MAX_KARAKTERA = 500

# odgovor
BROJ_SLIKA_PO_VRSTI = 3
GEMINI_MODEL = "gemini-3.5-flash-lite"

PORUKA_NEMA_REZULTATA = "The plant could not be identified. Please try again with a better photo and/or description."
PORUKA_NEDOSTUPAN = "The system cannot process your request at the moment. Please try again later."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Učitavam modele na", DEVICE, "...")

model_slike, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_SLIKE)
model_slike = model_slike.to(DEVICE).eval()

model_teksta = SentenceTransformer(MODEL_TEKSTA, device=DEVICE)

srednji_vektor_upita = np.load(KB_DIR / "bge_srednji_vektori.npz")["srednji_vektor_val_upita"]

with open(KB_DIR / "knowledge_base.json", encoding="utf-8") as f:
    knowledge_base = json.load(f)
BROJ_VRSTA = len(knowledge_base)

qdrant = QdrantClient(url=QDRANT_URL, timeout=60)

if os.environ.get("GEMINI_API_KEY"):
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=30_000))   # 30 s po zahtevu
else:
    gemini = None
    print("Nema GEMINI_API_KEY — zahtevi se neće obrađivati dok se ključ ne postavi.")


def greska(poruka):
    raise HTTPException(status_code=400, detail=poruka)


def proveri_slike(fajlovi):
    if len(fajlovi) > MAX_SLIKA:
        greska(f"Too many photos: {len(fajlovi)}. The maximum is {MAX_SLIKA}.")

    slike = []
    for naziv, sadrzaj in fajlovi:
        if len(sadrzaj) > MAX_MB * 1024 * 1024:
            greska(f"{naziv} is larger than {MAX_MB} MB.")

        try:
            slika = Image.open(io.BytesIO(sadrzaj))
            slika.load()
        except Exception:
            greska(f"{naziv} could not be opened as an image. Use JPEG, PNG or WebP.")

        if slika.format not in DOZVOLJENI_FORMATI:
            greska(f"{naziv} has an unsupported format ({slika.format}). Use JPEG, JPG OR PNG.")
        if min(slika.size) < MIN_STRANICA:
            greska(f"{naziv} is too small ({slika.size[0]} x {slika.size[1]} px). "
                   f"The shorter side must be at least {MIN_STRANICA} px.")

        slike.append(slika.convert("RGB"))
    return slike

@torch.no_grad()
def vektori_slika(slike):
    tenzori = torch.stack([preprocess_val(slika) for slika in slike])   # type: ignore[operator]
    vektori = model_slike.encode_image(tenzori.to(DEVICE))   # type: ignore[operator]
    return F.normalize(vektori, dim=-1).cpu().numpy()


def slikovna_grana(slike):
    if len(slike) == 0:
        return {"ishod": "no_result", "rang": []}
    skor_vrste = {}
    skor_poslednjeg_suseda = 0.0
    for vektor in vektori_slika(slike):
        odgovor = qdrant.query_points(collection_name=KOLEKCIJA_SLIKE, query=vektor.tolist(),
                                      limit=K, with_payload=["specie_id"])
        for point in odgovor.points:
            specie_id = (point.payload or {})["specie_id"]
            if point.score > skor_vrste.get(specie_id, -1.0):
                skor_vrste[specie_id] = point.score
        skor_poslednjeg_suseda = max(skor_poslednjeg_suseda, odgovor.points[-1].score)

    rang = sorted(skor_vrste, key=lambda s: skor_vrste[s], reverse=True)
    top1_score = skor_vrste[rang[0]]

    if len(rang) >= 2:
        gap = top1_score - skor_vrste[rang[1]]
    else:
        gap = top1_score - skor_poslednjeg_suseda  # ako je u 200 pointa samo jedna vrsta

    if top1_score < PRAG_ABSTAIN_SLIKA:
        ishod = "abstain"
    elif gap >= TAU_SLIKA:
        ishod = "siguran"
    else:
        ishod = "nesiguran"
    return {"ishod": ishod, "rang": rang, "top1_score": top1_score, "gap": gap}

def centriraj(vektor, srednji_vektor):
    vektor = vektor - srednji_vektor
    return vektor / np.linalg.norm(vektor)


def skor_po_vrsti(vektor_upita, nivo):
    odgovor = qdrant.query_points_groups(
        collection_name=KOLEKCIJA_TEKST,
        query=vektor_upita.tolist(),
        using="centriran",
        query_filter=Filter(must=[FieldCondition(key="nivo", match=MatchValue(value=nivo))]),
        group_by="specie_id",
        limit=BROJ_VRSTA,   # sve vrste
        group_size=1,       # najbolji chunk svake vrste
        with_payload=False,
    )
    skorovi = {}
    for grupa in odgovor.groups:
        skorovi[grupa.id] = grupa.hits[0].score
    return skorovi


def sirov_skor(sirovi_vektor_upita):
    odgovor = qdrant.query_points(
        collection_name=KOLEKCIJA_TEKST,
        query=sirovi_vektor_upita.tolist(),
        using="sirov",
        query_filter=Filter(must=[FieldCondition(key="nivo", match=MatchValue(value="prozor"))]),
        limit=1,
        with_payload=False,
    )
    return odgovor.points[0].score


def tekstualna_grana(upit):
    if upit == "":
        return {"ishod": "no_result", "rang": []}

    sirovi_vektor = model_teksta.encode(PREFIKS_UPITA + upit, normalize_embeddings=True)

    sirov = sirov_skor(sirovi_vektor)
    if sirov < PRAG_ABSTAIN_TEKST:
        return {"ishod": "abstain", "rang": [], "sirov_score": sirov}

    centriran_vektor = centriraj(sirovi_vektor, srednji_vektor_upita)
    # prozor i dokument se kombinuju
    prozor = skor_po_vrsti(centriran_vektor, "prozor")
    dokument = skor_po_vrsti(centriran_vektor, "dokument")

    skor_vrste = {}
    for specie_id in prozor:
        skor_vrste[specie_id] = (1 - TEZINA_DOKUMENTA) * prozor[specie_id] + TEZINA_DOKUMENTA * dokument[specie_id]
    rang = sorted(skor_vrste, key=lambda s: skor_vrste[s], reverse=True)

    gap = skor_vrste[rang[0]] - skor_vrste[rang[1]]
    if gap >= TAU_TEKST:
        ishod = "siguran"
    else:
        ishod = "nesiguran"
    return {"ishod": ishod, "rang": rang, "sirov_score": sirov, "gap": gap}

# ishod: no_result / abstain / siguran / nesiguran

def rrf(rang_slika, rang_tekst):
    poeni = {}
    for mesto, specie_id in enumerate(rang_slika, start=1):
        novi_poeni = W_SLIKA / (K_RRF + mesto)
        poeni[specie_id] = poeni.get(specie_id, 0.0) + novi_poeni
    for mesto, specie_id in enumerate(rang_tekst, start=1):
        novi_poeni = (1 - W_SLIKA) / (K_RRF + mesto)
        poeni[specie_id] = poeni.get(specie_id, 0.0) + novi_poeni
    return sorted(poeni, key=lambda s: poeni[s], reverse=True)


def fuzija(slika, tekst):
    bez_slike = slika["ishod"] in ("abstain", "no_result")
    bez_teksta = tekst["ishod"] in ("abstain", "no_result")

    if bez_slike and bez_teksta:
        return {"tip": "no_result", "vrste": []}

    if bez_slike or bez_teksta:
        grana = slika if bez_teksta else tekst
        if grana["ishod"] == "siguran":
            return {"tip": "siguran", "vrste": grana["rang"][:1]}
        return {"tip": "nesiguran", "vrste": grana["rang"][:DUZINA_LISTE]}

    top1_slika = slika["rang"][0]
    top1_tekst = tekst["rang"][0]

    if slika["ishod"] == "siguran" and tekst["ishod"] == "siguran":
        if top1_slika == top1_tekst:
            return {"tip": "siguran", "vrste": [top1_slika]}
        return {"tip": "nesiguran", "vrste": [top1_slika, top1_tekst]}

    if slika["ishod"] == "siguran" and top1_slika in tekst["rang"][:DUZINA_LISTE]:
        return {"tip": "siguran", "vrste": [top1_slika]}

    if tekst["ishod"] == "siguran" and top1_tekst in slika["rang"][:DUZINA_LISTE]:
        return {"tip": "siguran", "vrste": [top1_tekst]}

    return {"tip": "nesiguran", "vrste": rrf(slika["rang"], tekst["rang"])[:DUZINA_LISTE]}

def slike_za_vrstu(specie_id):
    # uvek iste slike: prve tri po imenu fajla
    fajlovi = sorted((SLIKE_DIR / specie_id).glob("*.jpg"))[:BROJ_SLIKA_PO_VRSTI]
    return [f"/slike/{specie_id}/{fajl.name}" for fajl in fajlovi]


SYSTEM_PROMPT = """You write the final answer of a plant identification system.

The system has already decided which species to show and in which order. You never choose, add,
remove or reorder species. You only write text.

You receive: the result type, and for every species its name and a reference text.

Write:
1. "uvod": one or two sentences that tell the user the result.
   - Result type "confident": the plant was identified as that species.
   - Result type "uncertain": the system is not sure; these are the most likely species, in order
     from the most likely.
2. For every species, "opis": two to four sentences on how to recognise the plant (flowers, leaves,
   stem, size). Use only facts from that species' reference text. Do not add anything from your own
   knowledge. If the text is very short, write less.

Write in plain English for a person who is not a botanist. Return the species in the same order and
with the same "specie_id" values you received. Answer only with JSON."""

SEMA_ODGOVORA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "uvod": types.Schema(type=types.Type.STRING),
        "vrste": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "specie_id": types.Schema(type=types.Type.STRING),
                    "opis": types.Schema(type=types.Type.STRING),
                },
                required=["specie_id", "opis"],
            ),
        ),
    },
    required=["uvod", "vrste"],
)


def naziv_vrste(specie_id):
    return knowledge_base[specie_id]["scientific_name"]


def generisanje_odgovora_LLM(odluka):
    tip = "confident" if odluka["tip"] == "siguran" else "uncertain"
    delovi = [f"Result type: {tip}"]
    for specie_id in odluka["vrste"]:
        delovi.append(f"\nSpecies (specie_id {specie_id}): {naziv_vrste(specie_id)}\n"
                      f"Reference text: {knowledge_base[specie_id]['tekst']}")
    poruka = "\n".join(delovi)

    try:
        assert gemini is not None   # bez ključa se ne stiže dovde
        odgovor = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=poruka,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=SEMA_ODGOVORA,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        response = json.loads(odgovor.text or "")
        opisi = {}
        for stavka in response["vrste"]:
            opisi[stavka["specie_id"]] = stavka["opis"]
        if set(opisi) != set(odluka["vrste"]) or not response["uvod"]:
            raise ValueError("LLM je vratio druge vrste ili prazan uvod")
        return response["uvod"], opisi
    except Exception as e:
        print(f"Pisanje odgovora nije uspelo: {e}")
        raise HTTPException(status_code=503, detail=PORUKA_NEDOSTUPAN)

app = FastAPI(title="Plant identification")

# frontend je na drugoj adresi (Vite, 5173)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# slike iz baze: /slike/<specie_id>/<ime fajla>
app.mount("/slike", StaticFiles(directory=SLIKE_DIR), name="slike")


def odluka_za_upit(opis, slike):
    """Sve do odluke, bez LLM-a."""
    opis = opis.strip()
    if opis == "" and len(slike) == 0:
        greska("Add a description or at least one photo.")
    if len(opis) > MAX_KARAKTERA:
        greska(f"The description is too long ({len(opis)} characters). The maximum is {MAX_KARAKTERA}.")

    fajlovi = []
    for fajl in slike:
        fajlovi.append((fajl.filename, fajl.file.read()))
    otvorene_slike = proveri_slike(fajlovi)

    rezultat_slike = slikovna_grana(otvorene_slike)
    rezultat_teksta = tekstualna_grana(opis)

    odluka = fuzija(rezultat_slike, rezultat_teksta)
    return odluka, rezultat_slike, rezultat_teksta


@app.post("/odluka")
def odluka_endpoint(opis: str = Form(""), slike: list[UploadFile] = File(default=[])):
    """Samo za evaluaciju (nb 10): odluka bez Gemini-ja."""
    odluka, rezultat_slike, rezultat_teksta = odluka_za_upit(opis, slike)
    return {
        "tip": odluka["tip"],
        "vrste": odluka["vrste"],
        "grane": {"slika": rezultat_slike["ishod"], "tekst": rezultat_teksta["ishod"]},
    }


@app.post("/identify")
def identify(opis: str = Form(""), slike: list[UploadFile] = File(default=[])):
    if gemini is None:
        raise HTTPException(status_code=503, detail=PORUKA_NEDOSTUPAN)

    odluka, rezultat_slike, rezultat_teksta = odluka_za_upit(opis, slike)

    odgovor = {
        "tip": odluka["tip"],
        "uvod": PORUKA_NEMA_REZULTATA,
        "vrste": [],
        "grane": {"slika": rezultat_slike["ishod"], "tekst": rezultat_teksta["ishod"]},
    }

    if odluka["tip"] != "no_result":
        uvod, opisi = generisanje_odgovora_LLM(odluka)

        odgovor["uvod"] = uvod
        for specie_id in odluka["vrste"]:
            odgovor["vrste"].append({
                "specie_id": specie_id,
                "naziv": naziv_vrste(specie_id),
                "opis": opisi[specie_id],
                "izvor": knowledge_base[specie_id]["izvor"],
                "slike": slike_za_vrstu(specie_id),
            })

    return odgovor


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
