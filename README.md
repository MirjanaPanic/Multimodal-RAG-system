# Multimodal RAG system for plant identification

Bachelor thesis project. The system identifies a plant species from up to 5 photos and/or a free-text
description of how the plant looks, and answers with one species or a short list of candidates.
The knowledge base covers 324 species.

- **Image branch:** BioCLIP embeds the photos and Qdrant finds the 200 nearest training images.
  The branch then decides on its own whether it is confident, uncertain, or abstains.
- **Text branch:** bge-large-en-v1.5 embeds the description (centred vectors) and Qdrant searches
  two-sentence windows plus whole species descriptions. It makes the same kind of decision.
- **Fusion:** a rule table combines the two decisions. Weighted RRF is used only to build the
  candidate list.
- **LLM:** Gemini only rewrites the already chosen species and their knowledge-base text into a
  readable answer. It never picks the species.

## Requirements

- Python 3.14, NVIDIA GPU with ~3 GB (CPU also works, but slowly)
- Docker (for Qdrant)
- Node 24
- A Gemini API key in `GEMINI_API_KEY`

```
pip install -r requirements.txt
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.1
```

`requirements.txt` lists the exact versions and model revisions. It also marks which packages are needed
by the application and which only by validation and evaluation.

