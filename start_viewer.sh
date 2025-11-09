#!/bin/bash
# Script per avviare il viewer web del vector database

cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run streamlit_app.py
