"""Hubness histogrami i UMAP za plant_text, sirovi vs centrirani -> hubness_histogram.png, hubness_umap.png; treba Qdrant"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from scipy.stats import skew
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

KOLEKCIJA = "plant_text"
VEKTORI = ["sirov", "centriran"]
K = 10

klijent = QdrantClient("localhost", port=6333)
tacke, _ = klijent.scroll(
    KOLEKCIJA, limit=5000, with_vectors=VEKTORI,
    scroll_filter=Filter(must=[FieldCondition(key="nivo", match=MatchValue(value="dokument"))]),
)
print(f"Ucitano tacaka: {len(tacke)}")


def broj_pojavljivanja(X, k=K):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)  # izbaci samu sebe
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return np.bincount(idx.ravel(), minlength=len(X))


rezultati = {}
for ime in VEKTORI:
    X = np.array([t.vector[ime] for t in tacke])
    Nk = broj_pojavljivanja(X)
    rezultati[ime] = (X, Nk)
    print(f"{ime}: skewness={skew(Nk):.2f}, max N_k={Nk.max()}, "
          f"N_k=0: {(Nk == 0).mean():.1%}, hubovi (>mean+2std): {(Nk > Nk.mean() + 2 * Nk.std()).mean():.1%}")

gornja_granica = max(Nk.max() for _, Nk in rezultati.values())
fig, ose = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
for osa, ime in zip(ose, VEKTORI):
    Nk = rezultati[ime][1]
    osa.hist(Nk, bins=np.arange(0, gornja_granica + 2))
    osa.set_yscale("log")
    osa.set_title(f"{ime}: skewness={skew(Nk):.2f}, N_k=0: {(Nk == 0).mean():.0%}")
    osa.set_xlabel(f"N_k (koliko puta je tacka medju {K} suseda)")
ose[0].set_ylabel("broj tacaka (log)")
plt.tight_layout()
plt.savefig("hubness_histogram.png", dpi=150)

# ista skala boja na oba grafika
fig, ose = plt.subplots(1, 2, figsize=(12, 5))
for osa, ime in zip(ose, VEKTORI):
    X, Nk = rezultati[ime]
    xy = umap.UMAP(random_state=0, metric="cosine").fit_transform(X)
    sc = osa.scatter(xy[:, 0], xy[:, 1], c=Nk, s=8 + 6 * Nk, cmap="viridis", vmin=0, vmax=gornja_granica)
    osa.set_title(ime)
    osa.set_xticks([]); osa.set_yticks([])
fig.colorbar(sc, ax=ose, label="N_k")
plt.savefig("hubness_umap.png", dpi=150)
print("Sacuvano: hubness_histogram.png, hubness_umap.png")
