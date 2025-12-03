"""
Feedback Components Module

Provides user feedback components including toasts, error displays,
empty states, and loading indicators.
"""

import streamlit as st
from typing import Optional, Callable, List
from ..theme import ICONS, COLORS


def show_success_toast(message: str, icon: str = "✅") -> None:
    """
    Display a success toast notification.
    
    Args:
        message: Success message to display
        icon: Optional icon override
    """
    st.toast(f"{icon} {message}", icon=icon)


def show_error_toast(message: str, icon: str = "❌") -> None:
    """
    Display an error toast notification.
    
    Args:
        message: Error message to display
        icon: Optional icon override
    """
    st.toast(f"{icon} {message}", icon=icon)


def show_error_with_details(
    message: str,
    details: Optional[str] = None,
    show_expander: bool = True
) -> None:
    """
    Display an error message with optional expandable details.
    
    Args:
        message: Main error message
        details: Optional detailed error information
        show_expander: Whether to show details in expander
    """
    st.error(f"{ICONS['error']} {message}")
    
    if details and show_expander:
        with st.expander("View error details"):
            st.code(details, language="text")
    elif details:
        st.code(details, language="text")


def show_warning_with_action(
    message: str,
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
) -> bool:
    """
    Display a warning message with optional action button.
    
    Args:
        message: Warning message
        action_label: Optional action button label
        action_callback: Optional callback function
    
    Returns:
        True if action button was clicked
    """
    st.warning(f"{ICONS['warning']} {message}")
    
    if action_label:
        if st.button(action_label, key=f"warning_action_{hash(message)}"):
            if action_callback:
                action_callback()
            return True
    return False


def show_empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_page: Optional[str] = None
) -> None:
    """
    Display an empty state with optional action.
    
    Args:
        title: Main title text
        message: Descriptive message
        icon: Icon to display
        action_label: Optional action button label
        action_page: Page to navigate to on action
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 40px;">
        <div style="font-size: 48px;">{icon}</div>
        <h3>{title}</h3>
        <p style="color: gray;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if action_label and action_page:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(action_label, use_container_width=True, type="primary"):
                st.session_state.current_page = action_page
                st.rerun()


def show_loading_status(
    phases: List[str],
    current_phase: int,
    message: str = "Processing..."
) -> None:
    """
    Display a multi-phase loading status.
    
    Args:
        phases: List of phase names
        current_phase: Current phase index (0-based)
        message: Status message
    """
    phase_icons = ["📄", "🧠", "💾", "✅"]
    
    # Phase indicator row
    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        icon = phase_icons[i] if i < len(phase_icons) else "⏳"
        with cols[i]:
            if i < current_phase:
                st.markdown(f"✅ {phase}")
            elif i == current_phase:
                st.markdown(f"**⏳ {phase}**")
            else:
                st.markdown(f"⬜ {phase}")
    
    # Progress info
    st.caption(message)


def show_info_callout(
    message: str,
    icon: str = "💡"
) -> None:
    """
    Display an informational callout.
    
    Args:
        message: Info message
        icon: Optional icon override
    """
    st.info(f"{icon} {message}")


def show_confirmation_dialog(
    title: str,
    message: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    warning: bool = True
) -> Optional[bool]:
    """
    Display a confirmation dialog.
    
    Args:
        title: Dialog title
        message: Confirmation message
        confirm_label: Confirm button label
        cancel_label: Cancel button label
        warning: Show as warning style
    
    Returns:
        True if confirmed, False if cancelled, None if no action
    """
    if warning:
        st.warning(f"⚠️ {title}")
    else:
        st.info(f"ℹ️ {title}")
    
    st.write(message)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"❌ {cancel_label}", use_container_width=True):
            return False
    with col2:
        if st.button(f"✅ {confirm_label}", use_container_width=True, type="primary"):
            return True
    
    return None


def show_processing_errors(
    errors: List[str],
    title: str = "Processing Errors"
) -> None:
    """
    Display a list of processing errors.
    
    Args:
        errors: List of error messages
        title: Section title
    """
    if not errors:
        return
    
    st.error(f"{ICONS['error']} {title}")
    
    for i, error in enumerate(errors, 1):
        with st.expander(f"Error {i}", expanded=False):
            st.code(error, language="text")


def show_success_summary(
    title: str,
    stats: dict,
    show_details: bool = True
) -> None:
    """
    Display a success summary with statistics.
    
    Args:
        title: Summary title
        stats: Dictionary of statistics
        show_details: Whether to show detailed stats
    """
    st.success(f"{ICONS['success']} {title}")
    
    if show_details and stats:
        cols = st.columns(len(stats))
        for i, (key, value) in enumerate(stats.items()):
            with cols[i]:
                st.metric(key, value)
