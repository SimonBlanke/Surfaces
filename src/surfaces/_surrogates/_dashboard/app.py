# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate Dashboard - Main Streamlit Application.

This is the entry point for the Streamlit dashboard.
Run with: streamlit run app.py
"""

import streamlit as st

from surfaces._surrogates._dashboard.database import get_dashboard_stats, init_db
from surfaces._surrogates._dashboard.sync import sync_all

# Page config
st.set_page_config(
    page_title="Surrogate Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for status colors
st.markdown(
    """
<style>
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-missing { color: #dc3545; font-weight: bold; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    """Main dashboard entry point."""
    st.title("Surrogate Dashboard")
    st.caption("Management UI for ML test function surrogates")

    # Initialize and sync on first load
    if "initialized" not in st.session_state:
        with st.spinner("Initializing database and syncing..."):
            init_db()
            sync_stats = sync_all()
            st.session_state.initialized = True
            st.session_state.sync_stats = sync_stats

    # Sidebar with stats
    with st.sidebar:
        st.header("Quick Stats")

        stats = get_dashboard_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Functions", stats["total_functions"])
            st.metric("Validations", stats["total_validations"])
        with col2:
            st.metric("With Surrogate", stats["with_surrogate"])
            st.metric("Trainings", stats["total_trainings"])

        # Coverage bar
        if stats["total_functions"] > 0:
            coverage = stats["with_surrogate"] / stats["total_functions"]
            st.progress(coverage, text=f"Coverage: {coverage:.0%}")

        st.divider()

        # Manual sync button
        if st.button("Sync Database", use_container_width=True):
            with st.spinner("Syncing..."):
                sync_stats = sync_all()
                st.session_state.sync_stats = sync_stats
            st.success(f"Synced {sync_stats['synced']} functions")
            st.rerun()

        st.divider()
        st.caption("Surfaces v0.5.1")
        st.caption("[GitHub](https://github.com/SimonBlanke/Surfaces)")

    # Main content - tabs for different pages
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Training",
            "Validation",
            "Details",
        ]
    )

    with tab1:
        from surfaces._surrogates._dashboard._pages import overview

        overview.render()

    with tab2:
        from surfaces._surrogates._dashboard._pages import training

        training.render()

    with tab3:
        from surfaces._surrogates._dashboard._pages import validation

        validation.render()

    with tab4:
        from surfaces._surrogates._dashboard._pages import details

        details.render()


if __name__ == "__main__":
    main()
