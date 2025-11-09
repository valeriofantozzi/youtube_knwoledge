# Vector Database Web Viewer

Interfaccia web interattiva per visualizzare e esplorare il vector database.

## Installazione

Le dipendenze sono già installate se hai eseguito `pip install -r requirements.txt`.

Se necessario:
```bash
source .venv/bin/activate
pip install streamlit plotly
```

## Avvio

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`

## Funzionalità

- **Statistiche**: Visualizza statistiche del database (documenti totali, video unici, ecc.)
- **Grafici**: Distribuzione dei chunks per data con grafici interattivi
- **Ricerca semantica**: Cerca nel database usando query in linguaggio naturale
- **Browser documenti**: Esplora i documenti con filtri per video ID e data
- **Lista video**: Visualizza tutti i video nel database con conteggio chunks

## Uso

1. Avvia l'applicazione con `streamlit run streamlit_app.py`
2. Usa la barra laterale per navigare
3. Inserisci una query nella sezione "Semantic Search" e clicca "Search"
4. Esplora i documenti usando i filtri nella sezione "Document Browser"
5. Visualizza la lista completa dei video nella sezione "Video List"

