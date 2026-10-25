"""Preuzima samo Description sekciju sa en Wikipedije, bez Similar species (wikipedia_description_raw.json)."""
import argparse
import csv
import json
import re
import time

import pandas as pd

from preuzmi_wikipedia import (
    CLANAK_URL, KB_DIR, MAPPING_CSV, PAUZA, napravi_sesiju, preuzmi_clanak,
)

RAW_JSON = KB_DIR / "wikipedia_description_raw.json"
NEMA_CSV = KB_DIR / "wikipedia_description_nema.csv"
IZVESTAJ_CSV = KB_DIR / "izvestaj_wikipedia_description.csv"

ZAGLAVLJE_RE = re.compile(r"(?m)^(={2,6}) (.+?) \1\s*$")


def izvuci_description(tekst):
    # sekcija traje do sledećeg zaglavlja istog ili višeg nivoa, podsekcije ulaze u nju
    zaglavlja = list(ZAGLAVLJE_RE.finditer(tekst))

    def kraj_sekcije(i):
        nivo = len(zaglavlja[i].group(1))
        for sledece in zaglavlja[i + 1:]:
            if len(sledece.group(1)) <= nivo:
                return sledece.start()
        return len(tekst)

    for i, z in enumerate(zaglavlja):
        if z.group(2).strip().lower() != "description":
            continue
        pocetak, kraj = z.end(), kraj_sekcije(i)

        # izbacuje se "Similar species"
        odsecci = []
        for j in range(i + 1, len(zaglavlja)):
            if zaglavlja[j].start() >= kraj:
                break
            if zaglavlja[j].group(2).strip().lower() in ("similar species", "similar taxa"):
                odsecci.append((zaglavlja[j].start(), kraj_sekcije(j)))
        delovi, tren = [], pocetak
        for od, do in sorted(odsecci):
            if od < tren:
                continue
            delovi.append(tekst[tren:od])
            tren = do
        delovi.append(tekst[tren:kraj])
        sadrzaj = "".join(delovi).strip()
        sadrzaj = re.sub(r"\n{3,}", "\n\n", sadrzaj)
        return sadrzaj or None
    return None


def upisi_rezultate(rezultati):
    KB_DIR.mkdir(parents=True, exist_ok=True)
    poredjani = dict(sorted(rezultati.items()))
    RAW_JSON.write_text(json.dumps(poredjani, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(IZVESTAJ_CSV, "w", encoding="utf-8", newline="") as f:
        p = csv.writer(f)
        p.writerow(["specie_id", "scientific_name", "status", "naslov_clanka", "duzina",
                    "izbacen_similar_species", "url"])
        for sid, z in poredjani.items():
            p.writerow([sid, z["scientific_name"], z["status"], z.get("naslov_clanka") or "",
                        len(z["tekst"]) if z.get("tekst") else 0,
                        "da" if z.get("izbacen_similar_species") else "",
                        z.get("url") or ""])

    with open(NEMA_CSV, "w", encoding="utf-8", newline="") as f:
        p = csv.writer(f)
        p.writerow(["specie_id", "scientific_name", "razlog", "url"])
        for sid, z in poredjani.items():
            if z["status"] != "ok":
                razlog = {"nema_clanka": "nema clanka na engleskoj Wikipediji",
                          "nema_description": "clanak postoji, ali nema Description",
                          "greska": "greska pri preuzimanju"}[z["status"]]
                p.writerow([sid, z["scientific_name"], razlog, z.get("url") or ""])


def main():
    parser = argparse.ArgumentParser(
        description="Preuzima samo Description pasus sa engleske Wikipedije.")
    parser.add_argument("--limit", type=int, help="obradi samo prvih N vrsta (za probu)")
    parser.add_argument("--ponovo", action="store_true", help="ignoriši ranije rezultate i keš")
    parser.add_argument("--bez-ssl-provere", action="store_true")
    args = parser.parse_args()

    mapping = pd.read_csv(MAPPING_CSV, dtype=str).dropna(subset=["scientific_name"])
    imena = dict(zip(mapping["specie_id"], mapping["scientific_name"].str.strip()))

    rezultati = {}
    if RAW_JSON.exists() and not args.ponovo:
        rezultati = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    # nema_clanka se ponavlja jer se neuspeo zahtev ne razlikuje od nepostojećeg članka
    gotovi = {i for i, z in rezultati.items() if z["status"] not in ("greska", "nema_clanka")}
    zaostali = [i for i in sorted(imena) if i not in gotovi]
    if args.limit:
        zaostali = zaostali[: args.limit]
    print(f"Za obradu: {len(zaostali)} od {len(imena)} vrsta\n")

    sesija = napravi_sesiju(args.bez_ssl_provere)
    try:
        for broj, sid in enumerate(zaostali, start=1):
            ime = imena[sid]
            naslov, tekst, iz_kesa = preuzmi_clanak(sesija, ime, "en", args.ponovo)
            if not iz_kesa:
                time.sleep(PAUZA)
            zapis = {"scientific_name": ime, "naslov_clanka": naslov, "url": None,
                     "status": "nema_clanka", "tekst": None}
            if naslov:
                zapis["url"] = CLANAK_URL.format("en", naslov.replace(" ", "_"))
            if tekst:
                opis = izvuci_description(tekst)
                if opis:
                    zapis["status"], zapis["tekst"] = "ok", opis
                    zapis["izbacen_similar_species"] = bool(
                        re.search(r"(?im)^=+ Similar (species|taxa) =+\s*$", tekst))
                else:
                    zapis["status"] = "nema_description"
            rezultati[sid] = zapis
            print(f"[{broj}/{len(zaostali)}] {sid} {ime}: {zapis['status']}"
                  + (f" ({len(zapis['tekst'])} znakova)" if zapis["tekst"] else "")
                  + (f"  [preusmereno na '{naslov}']" if naslov and naslov != ime else ""))
            if broj % 10 == 0:
                upisi_rezultate(rezultati)
    except KeyboardInterrupt:
        print("\nPrekinuto — snimam ono što je do sada skinuto.")
    finally:
        upisi_rezultate(rezultati)

    brojaci = {}
    for z in rezultati.values():
        brojaci[z["status"]] = brojaci.get(z["status"], 0) + 1
    print(f"\n=== Ukupno: {len(rezultati)} vrsta ===")
    for s, k in sorted(brojaci.items()):
        print(f"  {s}: {k}")
    print(f"Rezultat: {RAW_JSON}\nBez opisa: {NEMA_CSV}\nIzveštaj: {IZVESTAJ_CSV}")


if __name__ == "__main__":
    main()
