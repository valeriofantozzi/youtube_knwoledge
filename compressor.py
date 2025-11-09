#!/usr/bin/env python3
"""
Script per comprimere tutti i file .txt nella directory 2025 in un unico file.
Aggiunge separatori e metadata estratti dal filename.
"""

import os
import re
from pathlib import Path


def extract_metadata(filename):
    """
    Estrae metadata dal filename.
    Formato: YYYYMMDD_videoID_title.en.txt
    
    Returns:
        tuple: (date_formatted, title)
    """
    # Rimuove l'estensione .txt
    base_name = filename.replace('.txt', '')
    
    # Rimuove l'estensione .en se presente
    if base_name.endswith('.en'):
        base_name = base_name.replace('.en', '')
    
    # Pattern: YYYYMMDD_videoID_title
    # Estrae la data (primi 8 caratteri)
    date_str = base_name[:8]
    
    # Formatta la data come YYYY/MM/DD
    if len(date_str) == 8 and date_str.isdigit():
        date_formatted = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
    else:
        date_formatted = "Unknown"
    
    # Estrae il titolo: tutto dopo il secondo underscore
    parts = base_name.split('_', 2)
    if len(parts) >= 3:
        title = parts[2]
    else:
        title = base_name
    
    return date_formatted, title


def compress_files(input_dir, output_file):
    """
    Comprime tutti i file .txt nella directory input_dir in un unico file output_file.
    
    Args:
        input_dir: Directory contenente i file .txt
        output_file: File di output dove scrivere il contenuto compresso
    """
    input_path = Path(input_dir)
    
    # Trova tutti i file .txt e li ordina per filename
    txt_files = sorted(input_path.glob('*.txt'))
    
    if not txt_files:
        print(f"Nessun file .txt trovato in {input_dir}")
        return
    
    print(f"Trovati {len(txt_files)} file .txt")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, txt_file in enumerate(txt_files):
            print(f"Elaborando: {txt_file.name}")
            
            # Aggiunge separatore (tranne per il primo file)
            if i > 0:
                outfile.write('\n' + '=' * 80 + '\n\n')
            
            # Estrae metadata dal filename
            date_formatted, title = extract_metadata(txt_file.name)
            
            # Scrive metadata
            outfile.write(f"date: {date_formatted}\n")
            outfile.write(f"title: {title}\n")
            outfile.write('\n')
            
            # Legge e scrive il contenuto del file
            try:
                with open(txt_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    # Assicura che ci sia una newline alla fine se non c'è
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
            except Exception as e:
                print(f"Errore leggendo {txt_file.name}: {e}")
                outfile.write(f"[ERRORE: Impossibile leggere il file]\n")
    
    print(f"\nCompressione completata! File di output: {output_file}")


if __name__ == '__main__':
    # Directory di input
    input_directory = '2025'
    
    # File di output
    output_filename = 'compressed_output.txt'
    
    # Verifica che la directory esista
    if not os.path.exists(input_directory):
        print(f"Errore: La directory '{input_directory}' non esiste!")
        exit(1)
    
    # Esegue la compressione
    compress_files(input_directory, output_filename)

