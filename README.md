# 🧠 KnowBase — ricerca semantica immersiva per documenti

Un toolkit pratico e immediato per trasformare raccolte di documenti (SRT, PDF, TXT, Markdown...) in una knowledge base ricercabile semanticamente. Usa modelli di embedding diversi, collezioni isolate per modello e una UI web integrata per esplorare i risultati.

Per sviluppatori e power users: semplice da estendere, pensato per testare modelli e pipeline diverse senza rompere gli indici esistenti.

**✨ Highlights**

- 🤖 **Multi-model**: supporto per `BAAI/bge-large-en-v1.5` e `google/embeddinggemma-300m` (e altri tramite adapter)
- 🔐 **Collezioni isolate**: ogni modello scrive in collezioni separate in ChromaDB
- 🔄 **Pipeline modulare**: parsing → chunking → embeddings → store → retrieval
- 🎛️ **Interfacce**: script CLI per batch, API programmatica e interfaccia Streamlit per esplorazione

**⚡ Pronto per prototipi e sperimentazione**: caching dei modelli, selezione dinamica del device (CPU, CUDA, MPS), e helper per confronto di qualità tra modelli.

**🚀 Quick TL;DR (esempio rapido)**

1. 📦 Crea e attiva un virtualenv:

```
python -m venv .venv
source .venv/bin/activate
```

2. 📥 Installa dipendenze:

```
pip install -r requirements.txt
```

3. ⚙️ Processa file (default model impostato in `.env`):

```
python scripts/process_subtitles.py --input subtitles/ --output data/processed
```

4. 🔍 Cerca nei dati indicizzati:

```
python scripts/query_subtitles.py "come potrei rinvasare un'orchidea?"
```

5. 🌐 Avvia la UI:

```
./start_viewer.sh
```

**💡 Perché è figa?**

- ⚡ Cambio modello al volo: puoi confrontare embedding di modelli diversi senza mescolare i dati.
- 🔌 Facilmente estendibile: il pattern a adapter rende l'aggiunta di un nuovo modello minimale.
- ⏱️ Pensato per SRT e documenti con contesto temporale (subtitle-aware chunking).

**📁 Struttura chiave del repository**

- 🧠 `src/embeddings/` — adapter, loader e pipeline per generare embeddings.
- 🔤 `src/preprocessing/` — parser per SRT, chunker, normalizzazione testo.
- 🗄️ `src/vector_store/` — gestione ChromaDB, naming per collezioni model-specific.
- 🛠️ `scripts/` — script CLI per processare, migrare e interrogare il DB.
- 🎨 `streamlit_app.py` — interfaccia web per esplorare ricerche e cambiare modello.

**📌 Scorci pratici**

- 📚 Collezioni:
  - BGE: `document_embeddings_bge_large`
  - Gemma: `document_embeddings_gemma_300m`
- 📄 File utili: `requirements.txt`, `start_viewer.sh`, `scripts/process_subtitles.py`

📖 Vuoi andare oltre? Apri `USER_GUIDE.md` per istruzioni tecniche dettagliate, esempi di CLI e snippet per usare le pipeline dal codice Python.
