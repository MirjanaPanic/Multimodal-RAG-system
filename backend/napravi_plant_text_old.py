"""Kolekcija plant_text_old (bge-small, sirovi vektori), samo za poređenje hubness-a."""
import json
import math
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

BACKEND_DIR = Path(__file__).resolve().parent
KB_JSON = BACKEND_DIR / "knowledge_base" / "knowledge_base.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION = "plant_text_old"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENA = 500

with open(KB_JSON, encoding="utf-8") as f:
    knowledge_base = json.load(f)
print("Broj vrsta:", len(knowledge_base), "| DEVICE =", DEVICE)

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
tokenizer = model.tokenizer


def broj_tokena(tekst):
    return len(tokenizer.encode(tekst, add_special_tokens=False))


# chunkovanje iz nb 04
def ocisti_naslove(tekst):
    tekst = re.sub(r"\s*^=+[^\n]*=+\s*\Z", "", tekst, flags=re.MULTILINE)
    tekst = re.sub(r"^=+\s*(.+?)\s*=+[ \t]*\n\s*", r"\1: ", tekst, flags=re.MULTILINE)
    return tekst.strip()


KRAJ_RECENICE = r"(?<=[.!?])\s+(?=[A-Z])"
SEPARATORI = ["\n\n\n", "\n\n", "\n", KRAJ_RECENICE]


def podeli(tekst, separator):
    if separator == KRAJ_RECENICE:
        delovi = re.split(KRAJ_RECENICE, tekst)
    else:
        delovi = tekst.split(separator)
    return [deo.strip() for deo in delovi if deo.strip() != ""]


def podeli_na_chunkove(tekst, nivo=0):
    if broj_tokena(tekst) <= MAX_TOKENA:
        return [tekst]

    separator = SEPARATORI[nivo]
    celine = podeli(tekst, separator)
    if len(celine) == 1:
        return podeli_na_chunkove(tekst, nivo + 1)

    delovi = []
    for celina in celine:
        if broj_tokena(celina) <= MAX_TOKENA:
            delovi.append(celina)
        else:
            delovi += podeli_na_chunkove(celina, nivo + 1)

    spojnica = " " if separator == KRAJ_RECENICE else separator
    return spoji_u_chunkove(delovi, spojnica)


def spoji_u_chunkove(celine, spojnica):
    ukupno = sum(broj_tokena(celina) for celina in celine)
    broj_delova = math.ceil(ukupno / MAX_TOKENA)
    cilj = ukupno / broj_delova

    chunkovi = []
    tekuci = []
    tekuci_tokeni = 0
    for celina in celine:
        n = broj_tokena(celina)
        if len(tekuci) > 0:
            novo = tekuci_tokeni + n
            if novo > MAX_TOKENA or abs(novo - cilj) > abs(tekuci_tokeni - cilj):
                chunkovi.append(spojnica.join(tekuci))
                tekuci = []
                tekuci_tokeni = 0
        tekuci.append(celina)
        tekuci_tokeni += n
    chunkovi.append(spojnica.join(tekuci))

    if len(chunkovi) > 1 and broj_tokena(chunkovi[-2]) + broj_tokena(chunkovi[-1]) <= MAX_TOKENA:
        poslednji = chunkovi.pop()
        chunkovi[-1] = chunkovi[-1] + spojnica + poslednji
    return chunkovi


records = []
for specie_id in sorted(knowledge_base):
    zapis = knowledge_base[specie_id]
    chunkovi = podeli_na_chunkove(ocisti_naslove(zapis["tekst"]))
    for i, chunk in enumerate(chunkovi):
        records.append({
            "id": len(records),
            "payload": {
                "specie_id": specie_id,
                "scientific_name": zapis["scientific_name"],
                "nivo": "dokument",
                "chunk_index": i,
                "chunk_num": len(chunkovi),
                "text": chunk,
                "broj_tokena": broj_tokena(chunk),
                "izvor": zapis["izvor"],
            },
        })
print("Ukupno chunkova:", len(records))

embeddings = model.encode([r["payload"]["text"] for r in records], batch_size=BATCH_SIZE,
                          normalize_embeddings=True, show_progress_bar=True)

client = QdrantClient(url=QDRANT_URL)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config={"sirov": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
)
client.create_payload_index(COLLECTION, field_name="specie_id", field_schema=PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION, field_name="nivo", field_schema=PayloadSchemaType.KEYWORD)

for start in range(0, len(records), BATCH_SIZE):
    client.upsert(collection_name=COLLECTION, points=[
        PointStruct(id=records[i]["id"], vector={"sirov": embeddings[i].tolist()}, payload=records[i]["payload"])
        for i in range(start, min(start + BATCH_SIZE, len(records)))
    ])

print("Tacaka u kolekciji:", client.count(COLLECTION).count)
