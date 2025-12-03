"""
Metric Card Component

Provides styled metric display cards for statistics and KPIs.
"""

import streamlit as st
from typing import Optional, Union


def metric_card(
    label: str,
    value: Union[int, float, str],
    icon: str = "📊",
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
) -> None:
    """
    Display a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value
        icon: Icon to display
        delta: Optional change indicator
        delta_color: Color for delta ("normal", "inverse", "off")
        help_text: Optional help tooltip
    """
    st.metric(
        label=f"{icon} {label}",
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def metric_row(
    metrics: list,
    columns: int = 4
) -> None:
    """
    Display multiple metrics in a row.
    
    Args:
        metrics: List of metric dictionaries with keys:
                 label, value, icon (optional), delta (optional)
        columns: Number of columns
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                icon=metric.get("icon", "📊"),
                delta=metric.get("delta"),
                help_text=metric.get("help")
            )


def stat_box(
    label: str,
    value: Union[int, float, str],
    icon: str = "📊",
    bg_color: str = "#f8f9fa"
) -> None:
    """
    Display a styled stat box with custom background.
    
    Args:
        label: Stat label
        value: Stat value
        icon: Icon to display
        bg_color: Background color
    """
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    ">
        <div style="font-size: 24px;">{icon}</div>
        <div style="font-size: 28px; font-weight: bold;">{value}</div>
        <div style="color: gray; font-size: 14px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)
