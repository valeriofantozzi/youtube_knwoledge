# USER_GUIDE — Guida tecnica per KnowBase

Questa guida spiega come usare KnowBase a livello tecnico: setup, comandi CLI, API Python, gestione modelli e dettagli su come sono organizzate le collezioni di embedding.

**Requisiti**

- Python 3.9+ (consigliato 3.11)
- Virtualenv o venv
- (Opzionale) CUDA / MPS per accelerazione

## Configurazione ambiente

1. Crea e attiva ambiente virtuale:

```
python -m venv .venv
source .venv/bin/activate
```

2. Installa dipendenze:

```
pip install -r requirements.txt
```

3. Copia l'esempio `.env` e modifica le variabili principali:

```
cp .env.example .env
```

- `MODEL_NAME`: modello di default (es. `BAAI/bge-large-en-v1.5`)
- `DEVICE`: `cpu`, `cuda`, o `auto`
- `MODEL_CACHE_DIR`: directory per cache modelli

## Struttura essenziale del progetto

- `src/preprocessing/` — parsing (SRT), cleaning, chunking
- `src/embeddings/` — adapter dei modelli, gestione del caricamento e pipeline
- `src/vector_store/` — wrapper per ChromaDB, naming delle collezioni
- `scripts/` — utilità CLI per processare e interrogare

## Comandi CLI principali

- Processare una cartella di SRT (usando il modello di default):

```
python scripts/process_subtitles.py --input subtitles/ --output data/processed
```

- Processare con un modello specifico:

```
python scripts/process_subtitles.py --input subtitles/ --model "google/embeddinggemma-300m" --output data/processed
```

- Eseguire una query testuale tramite script CLI:

```
python scripts/query_subtitles.py "come curare un'orchidea?"
```

Opzioni comuni negli script (se esposte): `--input`, `--output`, `--model`, `--batch-size`, `--device`.

## Programmatic usage (API Python)

Esempio minimo per ottenere embeddings e salvarli:

```python
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.chroma_store import ChromaStore

# Inizializza pipeline con modello specifico
pipeline = EmbeddingPipeline(model_name="BAAI/bge-large-en-v1.5", device="auto")

# Genera embeddings per una lista di testi
texts = ["Testo 1", "Testo 2"]
embs = pipeline.embed_batch(texts)

# Salva su Chroma (nome collezione determinato dal modello)
store = ChromaStore(model_name=pipeline.model_name)
store.add_documents(texts, embs, metadatas=[{"source":"demo"}] * len(texts))
```

### Note su `EmbeddingPipeline`

- Gestisce tokenizzazione, chunking (se richiesto) e batching.
- Supporta caching dei modelli per evitare reload continui.

## Naming delle collezioni Chroma

- Convenzione: `document_embeddings_<model_tag>`
- Esempi:
  - BGE: `document_embeddings_bge_large`
  - Gemma: `document_embeddings_gemma_300m`

Questo evita conflitti tra dimensioni diverse e permette confronti incrociati senza sovrascrittura.

## Consigli pratici e troubleshooting

- Se vedi OOM su GPU: prova `DEVICE=cpu` o riduci `--batch-size`.
- Per test rapidi in locale: usa dimensioni ridotte o subset dei dati.
- Rimuovere collezioni: usare gli helper in `src/vector_store/` (attenzione: operazione distruttiva).

## Esempi avanzati

- Valutazione comparativa: processa lo stesso dataset con due modelli diversi e confronta retrieval score o qualitativamente i risultati nel `streamlit_app.py`.
- Embedding length mismatch: se confronti BGE(1024) con Gemma(768) tieni separate le collezioni e fai analisi dopo normalizzazione o proiezione esterna.

## Deploy rapido della UI

1. Assicurarsi che il DB vettoriale (Chroma) sia accessibile e persistente in `data/vector_db`.
2. Avviare la UI:

```
./start_viewer.sh
```

## Testing e sviluppo

- Esegui i test con `pytest`:

```
pytest -q
```

- Aggiungi test per nuove adapter nella cartella `tests/`.

## Contatti e risorse

Per estendere il supporto a nuovi modelli, aggiungi un adapter in `src/embeddings/adapters/` e registra il modello in `src/embeddings/model_registry.py`.

---

Se vuoi, posso aggiungere esempi di configurazione `.env`, o generare uno script di deploy Docker/Compose per eseguire la UI e il DB vettoriale in produzione.
