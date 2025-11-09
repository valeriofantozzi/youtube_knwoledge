#!/usr/bin/env python3
"""
Scarica tutti i sottotitoli dei video da un canale YouTube.

Logica:
  1) Estrae lista di tutti i video (URL) dal link del canale
  2) Per ogni video, scarica i sottotitoli in parallelo usando multiprocessing

Dipendenze:
  - yt-dlp  (pip install yt-dlp)
  - ffmpeg  (consigliato se vuoi convertire i sottotitoli in .srt)

Esempio uso:
  python3 download_youtube_channel_subtitles.py "https://www.youtube.com/@SomeChannel" \
      -o subtitles --format srt --include-shorts --limit 50

Note:
  - Se non specifichi "/videos" alla fine dell'URL del canale, lo aggiungeremo
    automaticamente per estrarre la lista dei video.
  - Per ottenere sia sottotitoli caricati dal creator che quelli automatici,
    vengono abilitati sia writesubtitles che writeautomaticsub.
  - Default: scarica SOLO i sottotitoli in inglese ("en"). Usa --all-langs
    per tutte le lingue oppure --langs per specificarne di diverse.
  - Con --format srt può essere necessario avere ffmpeg installato per la
    conversione da .vtt a .srt.
  - Il download usa multiprocessing per parallelizzare i download. Il numero
    di processi viene calcolato automaticamente in base ai core CPU disponibili.
    Puoi specificare manualmente il numero di worker con --workers.
"""

from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    import yt_dlp  # type: ignore
except Exception as e:  # pragma: no cover
    print("Errore: yt-dlp non è installato. Installa con: pip install yt-dlp", file=sys.stderr)
    raise


# ------------------------------
# Logging setup
# ------------------------------
LOGGER_NAME = "yt_subs_dl"
logger = logging.getLogger(LOGGER_NAME)


class YTDLPLogger:
    """Adattatore logger per yt-dlp (accetta debug/warning/error)."""

    def __init__(self, base_logger: logging.Logger) -> None:
        self._log = base_logger

    def debug(self, msg: str) -> None:  # yt-dlp chiama spesso .debug
        # Filtra le righe troppo verbose se necessario
        self._log.debug(msg)

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)


def setup_logging(verbosity: int) -> None:
    """Configura logging con livelli: 0=INFO, 1=DEBUG, >=2=TRACE-like."""
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Aumenta verbosità interna di yt-dlp se verbosity > 1
    if verbosity > 1:
        logging.getLogger("yt_dlp").setLevel(logging.DEBUG)


# ------------------------------
# URL helpers
# ------------------------------
def ensure_videos_tab(url: str) -> str:
    """Se è un URL canale YouTube generico, forza il tab /videos.

    Non modifica URL già puntati a /videos, /shorts, /streams, /watch, /playlist.
    """
    try:
        parsed = urlsplit(url)
    except Exception:
        logger.debug("URL non parseable, ritorno come dato: %s", url)
        return url

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")

    if "youtube.com" not in host:
        return url

    # Se punta già a una specifica sezione o a un contenuto, non tocchiamo.
    if any(seg in path for seg in ("/videos", "/shorts", "/streams", "/watch", "/playlist")):
        return url

    new_url = url.rstrip("/") + "/videos"
    logger.debug("Append /videos al canale: %s -> %s", url, new_url)
    return new_url


# ------------------------------
# Estrazione lista video
# ------------------------------
def _flatten_entries(info: Dict) -> Iterable[Dict]:
    """Appiattisce ricorsivamente le entries di yt-dlp in una lista di dict."""
    if not info:
        return []
    entries = info.get("entries")
    if entries is None:
        # Potrebbe essere un singolo URL dict
        return [info]
    flat: List[Dict] = []
    for e in entries:
        if not e:
            continue
        if isinstance(e, dict) and e.get("entries") is not None:
            flat.extend(list(_flatten_entries(e)))
        else:
            flat.append(e)
    return flat


def extract_video_list(
    channel_url: str,
    include_shorts: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Estrae info flat dei video dal canale/playlist.

    Ritorna una lista di dict con campi almeno: id, title, url (quando disponibile).
    """
    url = ensure_videos_tab(channel_url)
    ydl_opts = {
        "extract_flat": True,  # non scaricare metadata completi di ogni video, più veloce
        "skip_download": True,
        "noplaylist": False,
        "logger": YTDLPLogger(logger),
        "quiet": True,
        # Limita numero di item della playlist (se fornito)
        "playlistend": limit,
    }

    logger.info("Estrazione lista video da: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    logger.debug("Tipo radice estratta: %s | Keys: %s", type(info).__name__, list(info.keys())[:10])

    videos: List[Dict] = []
    for idx, e in enumerate(_flatten_entries(info)):
        vid_url = e.get("url") or e.get("webpage_url")
        vid_id = e.get("id")
        title = e.get("title")

        # Normalizza URL quando possibile
        if vid_url and not str(vid_url).startswith("http"):
            if isinstance(vid_url, str) and len(vid_url) == 11:  # probabile id YouTube
                vid_url = f"https://www.youtube.com/watch?v={vid_url}"
            else:
                vid_url = "https://www.youtube.com" + ("" if str(vid_url).startswith("/") else "/") + str(vid_url)

        # Filtra shorts se richiesto
        if not include_shorts and isinstance(vid_url, str) and "/shorts/" in vid_url:
            logger.debug("Skip shorts: %s | %s", vid_id, title)
            continue

        if not vid_url:
            logger.debug("Entry senza URL, skip: %s", e)
            continue

        videos.append({"id": vid_id, "title": title, "url": vid_url})
        if (idx + 1) % 50 == 0:
            logger.info("Raccolti %d video finora...", idx + 1)

        if limit is not None and len(videos) >= limit:
            break

    logger.info("Totale video estratti: %d", len(videos))
    return videos


# ------------------------------
# Download sottotitoli
# ------------------------------
def _check_subtitle_exists(
    video_id: Optional[str],
    output_dir: str,
    sub_format: str,
    langs: Optional[List[str]] = None,
) -> bool:
    """Controlla se i sottotitoli per un video sono già stati scaricati.
    
    Args:
        video_id: ID del video YouTube
        output_dir: Directory di output dove cercare
        sub_format: Formato sottotitoli (es. "srt")
        langs: Lista di codici lingua da controllare (None = qualsiasi lingua)
    
    Returns:
        True se almeno un file sottotitolo esiste già per questo video
    """
    if not video_id:
        return False
    
    if not os.path.exists(output_dir):
        return False
    
    # Cerca file che contengono l'ID del video nel nome
    # Pattern: *{video_id}*.{ext} in tutte le sottocartelle
    pattern = os.path.join(output_dir, "**", f"*{video_id}*.{sub_format}")
    existing_files = glob.glob(pattern, recursive=True)
    
    if not existing_files:
        return False
    
    # Se sono specificate lingue, controlla che almeno una corrisponda
    if langs:
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            # I file hanno formato: {date}_{id}_{title}.{lang}.{ext}
            # Controlla se contiene una delle lingue richieste
            for lang in langs:
                if f".{lang}." in filename or filename.endswith(f".{lang}.{sub_format}"):
                    return True
        # Nessuna lingua corrisponde, considera non scaricato
        return False
    
    # Nessuna lingua specificata, se esiste almeno un file è ok
    return True


def _calculate_optimal_workers() -> int:
    """Calcola il numero ottimale di worker processi basato sui core disponibili.
    
    Usa tutti i core disponibili, ma almeno 1 e al massimo il numero di CPU.
    """
    cpu_count = os.cpu_count() or 1
    # Usa tutti i core disponibili per massimizzare il throughput
    # Per operazioni I/O bound come il download, può essere utile usare anche più core
    optimal = cpu_count
    return max(1, optimal)


def _progress_hook(d: Dict) -> None:
    status = d.get("status")
    filename = d.get("filename") or d.get("info_dict", {}).get("_filename")
    if status == "finished":
        logger.info("Completato: %s", filename)
    elif status == "downloading":
        # Per i sottotitoli spesso non vengono emessi molti hook, ma logghiamo se avviene
        logger.debug("Scaricando: %s", filename)


def _download_single_video(
    video_data: Tuple[Dict, str, str, bool, Optional[List[str]]]
) -> Tuple[bool, str, bool]:
    """Scarica i sottotitoli per un singolo video.
    
    Args:
        video_data: Tupla contenente (video_dict, output_dir, sub_format, all_langs, langs)
    
    Returns:
        Tupla (success, video_url, skipped) dove:
        - success è True se il download è riuscito o è stato saltato
        - skipped è True se il file esisteva già
    """
    video, output_dir, sub_format, all_langs, langs = video_data
    
    vid_url = video["url"]
    vid_id = video.get("id")
    title = video.get("title")
    
    # Configura logger locale per questo processo
    # Il logger eredita la configurazione dal processo principale
    process_logger = logging.getLogger(LOGGER_NAME)
    
    # Controlla se il sottotitolo esiste già
    if _check_subtitle_exists(vid_id, output_dir, sub_format, langs):
        process_logger.info("⊘ [%s] %s - Già scaricato, skip", vid_id, title)
        return (True, vid_url, True)
    
    # Assicurati che il logger abbia almeno un handler (per compatibilità multiprocessing)
    # Questo è necessario perché su alcuni sistemi (es. Windows con spawn) i processi
    # figli non ereditano i handler del processo principale
    if not process_logger.handlers and process_logger.level == logging.NOTSET:
        # Solo se il logger non è configurato, aggiungi un handler di base
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S"))
        process_logger.addHandler(handler)
        process_logger.setLevel(logging.INFO)
    
    ydl_opts = {
        # Sottotitoli
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": sub_format,
        # Se true, prova tutte le lingue disponibili
        "allsubtitles": bool(all_langs),
        # Se vuoi limitare lingue specifiche, es. ["it", "en"]
        "subtitleslangs": langs or [],
        
        # Non scaricare il video, solo i sottotitoli
        "skip_download": True,
        
        # Naming file output
        "outtmpl": os.path.join(
            output_dir,
            "%(uploader)s",
            "%(upload_date)s_%(id)s_%(title)s.%(ext)s",
        ),
        "restrictfilenames": True,
        
        # Robustezza
        "ignoreerrors": True,
        
        # Logging
        "logger": YTDLPLogger(process_logger),
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # yt-dlp gestisce internamente l'esistenza dei file di sottotitoli
            retcode = ydl.download([vid_url])
            # retcode 0 = ok
            if retcode == 0:
                process_logger.info("✓ [%s] %s", vid_id, title)
                return (True, vid_url, False)
            else:
                process_logger.warning("✗ [%s] %s - yt-dlp ritornato codice %s", vid_id, title, retcode)
                return (False, vid_url, False)
    except Exception as e:
        process_logger.exception("✗ [%s] %s - Errore: %s", vid_id, title, e)
        return (False, vid_url, False)


def download_subtitles(
    videos: List[Dict],
    output_dir: str,
    sub_format: str = "srt",
    all_langs: bool = False,
    langs: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, int]:
    """Scarica sottotitoli per ogni video usando multiprocessing.

    Ritorna un dizionario con conteggi di successi/errori/saltati.
    
    Args:
        videos: Lista di dizionari con info video
        output_dir: Directory di output
        sub_format: Formato sottotitoli
        all_langs: Scarica tutte le lingue disponibili
        langs: Lista codici lingua specifici
        num_workers: Numero di processi worker (None = auto-calcolato)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calcola numero ottimale di worker se non specificato
    if num_workers is None:
        num_workers = _calculate_optimal_workers()
    
    # Limita il numero di worker al numero di video disponibili
    num_workers = min(num_workers, len(videos))
    
    logger.info("Inizio download sottotitoli per %d video usando %d processi", len(videos), num_workers)
    logger.debug("Core disponibili: %d", os.cpu_count() or 1)

    # Prepara i dati per ogni video (tupla per passare a _download_single_video)
    video_tasks = [
        (video, output_dir, sub_format, all_langs, langs)
        for video in videos
    ]

    success = 0
    failures = 0
    skipped = 0

    # Usa multiprocessing solo se ci sono più video di quanti worker
    # Altrimenti usa un singolo processo per evitare overhead
    if num_workers > 1 and len(videos) > 1:
        # Usa multiprocessing.Pool per parallelizzare i download
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_download_single_video, video_tasks)
        
        # Conta successi, fallimenti e saltati
        for success_flag, vid_url, was_skipped in results:
            if was_skipped:
                skipped += 1
            elif success_flag:
                success += 1
            else:
                failures += 1
    else:
        # Fallback sequenziale per piccoli batch o singolo worker
        logger.debug("Uso modalità sequenziale (worker=%d, video=%d)", num_workers, len(videos))
        for video_task in video_tasks:
            success_flag, vid_url, was_skipped = _download_single_video(video_task)
            if was_skipped:
                skipped += 1
            elif success_flag:
                success += 1
            else:
                failures += 1

    logger.info("Download completato. Successi: %d | Saltati: %d | Errori: %d", success, skipped, failures)
    return {"success": success, "failures": failures, "skipped": skipped}


# ------------------------------
# CLI
# ------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scarica i sottotitoli di tutti i video di un canale YouTube",
    )
    p.add_argument("channel_url", help="URL del canale YouTube (es. https://www.youtube.com/@nome)")
    p.add_argument(
        "-o",
        "--output-dir",
        default="subtitles",
        help="Directory di output (default: subtitles)",
    )
    p.add_argument(
        "--format",
        default="srt",
        choices=["srt", "vtt", "ass", "ttml"],
        help="Formato sottotitoli desiderato (default: srt)",
    )
    p.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Codici lingua desiderati (default: en). Esempio: --langs it en",
    )
    p.add_argument(
        "--all-langs",
        action="store_true",
        help="Scarica tutte le lingue disponibili (override di --langs)",
    )
    p.add_argument(
        "--include-shorts",
        action="store_true",
        help="Includi anche YouTube Shorts (default: esclusi)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita il numero massimo di video da processare (debug/test)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Aumenta verbosità log (-v=DEBUG, -vv=traccia interna)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Numero di processi worker per il download parallelo (default: auto-calcolato basato sui core CPU)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    logger.info("Avvio script con argomenti: %s", args)
    logger.debug("Versione yt-dlp: %s", getattr(yt_dlp, "version", "unknown"))

    # Default: solo inglese se non specificato
    langs = args.langs if args.langs else ["en"]
    all_langs = bool(args.all_langs)
    if all_langs:
        logger.info("Scarico tutte le lingue disponibili")
    else:
        logger.info("Lingue richieste: %s", ", ".join(langs))

    try:
        videos = extract_video_list(
            args.channel_url,
            include_shorts=args.include_shorts,
            limit=args.limit,
        )
    except Exception as e:
        logger.exception("Errore durante l'estrazione della lista dei video: %s", e)
        return 2

    if not videos:
        logger.error("Nessun video trovato. Controlla l'URL del canale o eventuali restrizioni.")
        return 3

    stats = download_subtitles(
        videos,
        output_dir=args.output_dir,
        sub_format=args.format,
        all_langs=all_langs,
        langs=langs,
        num_workers=args.workers,
    )

    # Exit code basato su presenza errori
    return 0 if stats.get("failures", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
