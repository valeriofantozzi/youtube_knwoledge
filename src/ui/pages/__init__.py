"""
Page modules for Streamlit application.

Each page is a separate module with a render function.
"""

from .load_documents import render_load_documents_page
from .postprocessing import render_postprocessing_page
from .search import render_search_page

__all__ = [
    "render_load_documents_page",
    "render_postprocessing_page",
    "render_search_page",
]
