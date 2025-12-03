"""
Theme Module

Defines the visual design system including colors, icons, and styling helpers
for the Streamlit application.
"""

from typing import Dict, Any, List, Union
import streamlit as st


# Color palette (Streamlit-compatible)
COLORS: Dict[str, Union[str, List[str]]] = {
    # Primary actions
    "primary": "#FF4B4B",           # Streamlit red - main CTAs
    "primary_hover": "#FF6B6B",
    
    # Status colors
    "success": "#28A745",           # Green - completed, positive
    "warning": "#FFC107",           # Yellow - caution, in-progress
    "error": "#DC3545",             # Red - errors, failures
    "info": "#17A2B8",              # Cyan - informational
    
    # Semantic colors for data visualization
    "cluster_palette": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ],
    
    # Similarity score gradient
    "score_high": "#28A745",        # > 0.8
    "score_medium": "#FFC107",      # 0.5 - 0.8
    "score_low": "#DC3545",         # < 0.5
    
    # Background colors
    "bg_light": "#F8F9FA",
    "bg_dark": "#343A40",
    "bg_card": "#FFFFFF",
}

# Icon set (emoji-based for Streamlit)
ICONS: Dict[str, str] = {
    # Navigation
    "load": "📥",
    "analysis": "🔬",
    "search": "🔍",
    "home": "🏠",
    
    # Status
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "loading": "⏳",
    "processing": "🔄",
    "info": "ℹ️",
    
    # Actions
    "run": "▶️",
    "download": "⬇️",
    "delete": "🗑️",
    "add": "➕",
    "edit": "✏️",
    "refresh": "🔄",
    "settings": "⚙️",
    
    # Data
    "chart": "📊",
    "metrics": "📈",
    "files": "📁",
    "video": "🎬",
    "document": "📄",
    "database": "🗃️",
    
    # Clustering
    "cluster": "🔬",
    "centroid": "🎯",
    "outlier": "📍",
    
    # Other
    "tip": "💡",
    "robot": "🤖",
    "calendar": "📅",
    "time": "⏱️",
    "speed": "⚡",
    "dimension": "🔢",
}

# Clustering presets for quick configuration
CLUSTERING_PRESETS: Dict[str, Dict[str, Any]] = {
    "🚀 Quick": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "description": "Fast results, may have more noise"
    },
    "⚖️ Balanced": {
        "min_cluster_size": 15,
        "min_samples": 5,
        "description": "Good balance of speed and quality"
    },
    "🔬 Detailed": {
        "min_cluster_size": 30,
        "min_samples": 10,
        "description": "Fewer, more coherent clusters"
    }
}


def get_score_color(score: float) -> str:
    """
    Get color based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
    
    Returns:
        Hex color string
    """
    if score >= 0.8:
        return str(COLORS["score_high"])
    elif score >= 0.5:
        return str(COLORS["score_medium"])
    else:
        return str(COLORS["score_low"])


def get_score_label(score: float) -> str:
    """
    Get quality label based on similarity score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
    
    Returns:
        Quality label string
    """
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.5:
        return "Good"
    else:
        return "Low"


def inject_custom_css() -> None:
    """
    Inject custom CSS styles into the Streamlit app.
    
    Should be called once at the beginning of the app.
    """
    custom_css = """
    <style>
    /* Metric card hover effects */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Result card styling */
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    
    /* Progress bar customization */
    div.stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    
    /* Button styling */
    .stButton > button {
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Container borders */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 8px;
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 500;
    }
    
    /* Info/warning/error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def format_number(num: int) -> str:
    """
    Format large numbers with K/M suffixes.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
