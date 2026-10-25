"""Preuzima Description sekcije sa Wikipedije za vrste bez WFO opisa (wikipedia_raw.json)."""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_DIR = BACKEND_DIR / "dataset"
KB_DIR = BACKEND_DIR / "knowledge_base"

MAPPING_CSV = DATASET_DIR / "specie_id - scientific_name mapping.csv"
WFO_RAW_JSON = KB_DIR / "wfo_raw.json"
WIKI_RAW_JSON = KB_DIR / "wikipedia_raw.json"
IZVESTAJ_CSV = KB_DIR / "izvestaj_wikipedia.csv"
WIKI_CACHE = KB_DIR / "_wiki_cache"

API = "https://{}.wikipedia.org/w/api.php"
CLANAK_URL = "https://{}.wikipedia.org/wiki/{}"
ALT_IMENA_CSV = KB_DIR / "alternativna_imena.csv"

PAUZA = 2.0        # sekundi između zahteva, inače 429
TAJMAUT = 30
POKUSAJA = 4
MAX_UZASTOPNIH_GRESAKA = 5
MIN_DUZINA = 60        # kraće od ovoga je beskorisno za pretragu
MIN_DUZINA_UVOD = 200  # kraći uvod ne nosi opis

# ostalo se koristi samo ako nema "Description"
ZAGLAVLJA = ["Description", "Morphology", "Botanical description", "Characteristics"]

# drugi jezici ako en nema opis, nemački prvi (sekcija Beschreibung sa merama)
JEZICI = [
    ("de", ["Beschreibung", "Merkmale"]),
    ("fr", ["Description", "Description morphologique", "Morphologie", "Caractéristiques"]),
    ("es", ["Descripción", "Morfología", "Características"]),
    ("it", ["Descrizione", "Morfologia", "Caratteristiche"]),
    ("ca", ["Descripció", "Morfologia"]),
    ("nl", ["Beschrijving", "Kenmerken"]),
    ("cs", ["Popis"]),
    ("hr", ["Opis"]),
    ("pt", ["Descrição", "Características"]),
    ("ja", ["特徴", "形態"]),
]

# WFO opis ovih vrsta je samo spisak praznih polja
VRSTE_PRAZNE_KARTICE = ["0217", "0262", "0428"]

# flore sa popunjenim formularom umesto rečenica, skida se i Wikipedia za poređenje
TELEGRAFSKI_IZVORI = ("Brazilian Flora",)


def napravi_sesiju(bez_ssl_provere=False):
    sesija = requests.Session()
    # Wikipedia traži User-Agent
    sesija.headers.update({
        "User-Agent": "diplomski-rad-RAG/1.0 (studentski projekat; kontakt preko GitHub-a)",
    })
    if bez_ssl_provere:
        sesija.verify = False
        requests.packages.urllib3.disable_warnings()
    return sesija


def kes_putanja(jezik, ime):
    # engleski je bez prefiksa zbog ranije skinutog keša
    bezbedno = re.sub(r"[^A-Za-z0-9._-]", "_", ime)
    return WIKI_CACHE / (f"{bezbedno}.json" if jezik == "en" else f"{jezik}__{bezbedno}.json")


def preuzmi_clanak(sesija, ime, jezik="en", ponovo=False):
    # vraća (naslov, tekst, iz_kesa); redirects=1 prati preusmerenje (npr. Valeriana rubra na Centranthus ruber)
    kes_fajl = kes_putanja(jezik, ime)
    if kes_fajl.exists() and not ponovo:
        podaci = json.loads(kes_fajl.read_text(encoding="utf-8"))
        return podaci.get("naslov"), podaci.get("tekst"), True

    parametri = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": ime,
    }
    for pokusaj in range(1, POKUSAJA + 1):
        try:
            odgovor = sesija.get(API.format(jezik), params=parametri, timeout=TAJMAUT)
            if odgovor.status_code == 429:   # 429, čeka se duže
                time.sleep(PAUZA * pokusaj * 5)
                continue
            if odgovor.status_code != 200:
                return None, None, False
            stranice = odgovor.json().get("query", {}).get("pages", {})
            if not stranice:
                return None, None, False
            stranica = next(iter(stranice.values()))
            if "missing" in stranica:
                naslov, tekst = None, None
            else:
                naslov, tekst = stranica.get("title"), stranica.get("extract")
            WIKI_CACHE.mkdir(parents=True, exist_ok=True)
            kes_fajl.write_text(
                json.dumps({"naslov": naslov, "tekst": tekst}, ensure_ascii=False),
                encoding="utf-8")
            return naslov, tekst, False
        except (requests.RequestException, ValueError, StopIteration):
            time.sleep(PAUZA * pokusaj * 2)
    return None, None, False


def jezicke_veze(sesija, ime):
    # {jezik: naslov} za jezike na kojima članak postoji
    parametri = {"action": "query", "format": "json", "prop": "langlinks",
                 "lllimit": "200", "redirects": "1", "titles": ime}
    try:
        odgovor = sesija.get(API.format("en"), params=parametri, timeout=TAJMAUT)
        if odgovor.status_code != 200:
            return {}
        stranice = odgovor.json().get("query", {}).get("pages", {})
        stranica = next(iter(stranice.values()))
        return {v["lang"]: v["*"] for v in (stranica.get("langlinks") or [])}
    except (requests.RequestException, ValueError, StopIteration, KeyError):
        return {}


def izvuci_sekciju(tekst, zaglavlja=None, dozvoli_uvod=True):
    # sekcija ide do sledećeg zaglavlja prvog nivoa, podsekcije su deo opisa (de Wikipedia)
    if not tekst:
        return None, None
    for zaglavlje in (zaglavlja if zaglavlja is not None else ZAGLAVLJA):
        pogodak = re.search(rf"(?m)^== {re.escape(zaglavlje)} ==\s*$", tekst)
        if not pogodak:
            continue
        ostatak = tekst[pogodak.end():]
        sledece = re.search(r"(?m)^== .+ ==\s*$", ostatak)
        sadrzaj = (ostatak[: sledece.start()] if sledece else ostatak).strip()
        if len(sadrzaj) >= MIN_DUZINA:
            return zaglavlje, sadrzaj

    # nema sekcije, uzima se uvod
    if dozvoli_uvod:
        pogodak = re.search(r"(?m)^== .+ ==\s*$", tekst)
        uvod = (tekst[: pogodak.start()] if pogodak else tekst).strip()
        if len(uvod) >= MIN_DUZINA_UVOD:
            return "uvod", uvod
    return None, None


def nadji_opis(sesija, ime, alt_imena, ponovo=False):
    # redom: en pod naučnim imenom, en pod alt imenom, drugi jezici, uvod en članka
    rezultat = {"naslov_clanka": None, "jezik": None, "url": None,
                "status": "nema_clanka", "sekcija": None, "tekst": None}
    en_naslov, en_tekst = None, None

    # en, pod svim imenima
    for kandidat in [ime] + alt_imena:
        naslov, tekst, iz_kesa = preuzmi_clanak(sesija, kandidat, "en", ponovo)
        if not iz_kesa:
            time.sleep(PAUZA)
        if not tekst:
            continue
        if en_tekst is None:
            en_naslov, en_tekst = naslov, tekst
        zaglavlje, sadrzaj = izvuci_sekciju(tekst, dozvoli_uvod=False)
        if sadrzaj:
            return {**rezultat, "naslov_clanka": naslov, "jezik": "en",
                    "url": CLANAK_URL.format("en", naslov.replace(" ", "_")),
                    "status": "ok", "sekcija": zaglavlje, "tekst": sadrzaj}

    # drugi jezici
    veze = {}
    for kandidat in ([en_naslov] if en_naslov else []) + [ime] + alt_imena:
        veze = jezicke_veze(sesija, kandidat)
        time.sleep(PAUZA)
        if veze:
            break

    for jezik, zaglavlja in JEZICI:
        naslov_na_jeziku = veze.get(jezik)
        if not naslov_na_jeziku and veze:
            continue           # postoji negde, ali ne na ovom jeziku
        if not naslov_na_jeziku:
            naslov_na_jeziku = ime   # nema veza, proba se naučno ime
        naslov, tekst, iz_kesa = preuzmi_clanak(sesija, naslov_na_jeziku, jezik, ponovo)
        if not iz_kesa:
            time.sleep(PAUZA)
        if not tekst:
            continue
        zaglavlje, sadrzaj = izvuci_sekciju(tekst, zaglavlja, dozvoli_uvod=False)
        if sadrzaj:
            return {**rezultat, "naslov_clanka": naslov, "jezik": jezik,
                    "url": CLANAK_URL.format(jezik, naslov.replace(" ", "_")),
                    "status": "ok", "sekcija": zaglavlje, "tekst": sadrzaj}

    # uvod en članka
    if en_tekst:
        zaglavlje, sadrzaj = izvuci_sekciju(en_tekst, [], dozvoli_uvod=True)
        if sadrzaj:
            return {**rezultat, "naslov_clanka": en_naslov, "jezik": "en",
                    "url": CLANAK_URL.format("en", en_naslov.replace(" ", "_")),
                    "status": "ok", "sekcija": "uvod", "tekst": sadrzaj}
        return {**rezultat, "naslov_clanka": en_naslov, "jezik": "en",
                "url": CLANAK_URL.format("en", en_naslov.replace(" ", "_")),
                "status": "nema_sekcije"}
    return rezultat


def ucitaj_alt_imena():
    # specie_id: [alternativna imena] iz alternativna_imena.csv
    if not ALT_IMENA_CSV.exists():
        return {}
    tabela = pd.read_csv(ALT_IMENA_CSV, dtype=str)
    alt = {}
    for _, red in tabela.iterrows():
        alt.setdefault(red["specie_id"], []).append(red["alternativno_ime"].strip())
    return alt


def upisi_rezultate(rezultati):
    KB_DIR.mkdir(parents=True, exist_ok=True)
    poredjani = dict(sorted(rezultati.items()))

    WIKI_RAW_JSON.write_text(
        json.dumps(poredjani, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(IZVESTAJ_CSV, "w", encoding="utf-8", newline="") as fajl:
        pisac = csv.writer(fajl)
        pisac.writerow([
            "specie_id", "scientific_name", "naslov_clanka", "jezik", "status",
            "sekcija", "duzina", "url",
        ])
        for specie_id, zapis in poredjani.items():
            pisac.writerow([
                specie_id, zapis["scientific_name"], zapis.get("naslov_clanka") or "",
                zapis.get("jezik") or "en", zapis["status"], zapis.get("sekcija") or "",
                len(zapis["tekst"]) if zapis.get("tekst") else 0,
                zapis.get("url") or "",
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Preuzima Description sekcije sa engleske Wikipedije (bez menjanja teksta).")
    parser.add_argument("--limit", type=int, help="obradi samo prvih N vrsta (za probu)")
    parser.add_argument("--vrste", nargs="+", help="obradi samo navedene specie_id")
    parser.add_argument("--ponovo", action="store_true", help="ignoriši ranije rezultate i keš")
    parser.add_argument("--bez-ssl-provere", action="store_true",
                        help="poslednje rešenje ako puca SSL greška (nesigurno)")
    args = parser.parse_args()

    if not WFO_RAW_JSON.exists():
        raise SystemExit(f"Nema {WFO_RAW_JSON} — prvo pokreni: python preuzmi_wfo.py")

    mapping = pd.read_csv(MAPPING_CSV, dtype=str).dropna(subset=["scientific_name"])
    imena = dict(zip(mapping["specie_id"], mapping["scientific_name"]))
    wfo = json.loads(WFO_RAW_JSON.read_text(encoding="utf-8"))
    alt_imena = ucitaj_alt_imena()

    # bez WFO opisa, prazne kartice i telegrafski izvori
    trebaju = [i for i, z in sorted(wfo.items()) if z["status"] != "ok"]
    trebaju += VRSTE_PRAZNE_KARTICE
    for specie_id, zapis in wfo.items():
        if zapis.get("izabrani") is None:
            continue
        izvor = zapis["opisi"][zapis["izabrani"]]["izvor"]
        if any(izvor.startswith(x) for x in TELEGRAFSKI_IZVORI):
            trebaju.append(specie_id)
    trebaju = sorted(set(trebaju) & set(imena))

    rezultati = {}
    if WIKI_RAW_JSON.exists() and not args.ponovo:
        rezultati = json.loads(WIKI_RAW_JSON.read_text(encoding="utf-8"))
        # ove se obrađuju ponovo, članak je u kešu pa nema novih zahteva
        gotovi = {i for i, z in rezultati.items()
                  if z["status"] not in ("greska", "nema_sekcije", "nema_clanka")}
        print(f"Nastavljam: {len(gotovi)} vrsta je već obrađeno.")
    else:
        gotovi = set()

    if args.vrste:
        # navedene vrste se obrađuju uvek, i kad imaju WFO opis
        trebaju = [i for i in args.vrste if i in imena]
        gotovi -= set(args.vrste)
    zaostali = [i for i in trebaju if i not in gotovi]
    if args.limit:
        zaostali = zaostali[: args.limit]

    print(f"Za obradu: {len(zaostali)} vrsta (ukupno bez WFO opisa: {len(trebaju)})\n")

    sesija = napravi_sesiju(args.bez_ssl_provere)
    uzastopnih_gresaka = 0
    try:
        for broj, specie_id in enumerate(zaostali, start=1):
            ime = imena[specie_id].strip()
            print(f"[{broj}/{len(zaostali)}] {specie_id}  {ime}")

            nadjeno = nadji_opis(sesija, ime, alt_imena.get(specie_id, []), args.ponovo)
            zapis = {"scientific_name": ime, **nadjeno}

            if zapis["status"] == "ok":
                uzastopnih_gresaka = 0
                print(f"    -> [{zapis['jezik']}] '{zapis['naslov_clanka']}': "
                      f"sekcija {zapis['sekcija']}, {len(zapis['tekst'])} znakova")
            elif zapis["status"] == "nema_sekcije":
                uzastopnih_gresaka = 0
                print(f"    -> '{zapis['naslov_clanka']}': nema morfološke sekcije")
            else:
                print("    -> nema članka ni na jednom jeziku")

            rezultati[specie_id] = zapis

            if broj % 5 == 0:
                upisi_rezultate(rezultati)
    except KeyboardInterrupt:
        print("\nPrekinuto sa tastature — snimam ono što je do sada skinuto.")
    finally:
        upisi_rezultate(rezultati)

    brojaci, po_sekciji = {}, {}
    for zapis in rezultati.values():
        brojaci[zapis["status"]] = brojaci.get(zapis["status"], 0) + 1
        if zapis.get("sekcija"):
            po_sekciji[zapis["sekcija"]] = po_sekciji.get(zapis["sekcija"], 0) + 1
    print(f"\n=== Ukupno u wikipedia_raw.json: {len(rezultati)} vrsta ===")
    for status, koliko in sorted(brojaci.items()):
        print(f"  {status}: {koliko}")
    if po_sekciji:
        print("  po sekciji:")
        for sekcija, koliko in sorted(po_sekciji.items(), key=lambda p: -p[1]):
            print(f"    {koliko:4d}  {sekcija}")
    print(f"\nRezultat: {WIKI_RAW_JSON}")
    print(f"Izveštaj: {IZVESTAJ_CSV}")


if __name__ == "__main__":
    main()
