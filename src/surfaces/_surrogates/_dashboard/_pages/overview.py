# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Overview page - Table of all ML functions with surrogate status.
"""

import pandas as pd
import streamlit as st

from surfaces._surrogates._dashboard.database import get_overview_data

# R2 threshold for "good" status
R2_THRESHOLD = 0.95


def get_status(row: dict) -> str:
    """Determine status based on surrogate existence and R2 score."""
    if not row["has_surrogate"]:
        return "Missing"
    if row["latest_r2"] is None:
        return "Not Validated"
    if row["latest_r2"] < R2_THRESHOLD:
        return "Needs Attention"
    return "Good"


def get_status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "Good": "#28a745",
        "Needs Attention": "#ffc107",
        "Not Validated": "#17a2b8",
        "Missing": "#dc3545",
    }
    return colors.get(status, "#6c757d")


def render():
    """Render the overview page."""
    st.header("Function Overview")

    # Get data
    data = get_overview_data()

    if not data:
        st.warning("No functions found. Run sync to populate the database.")
        return

    # Add status column
    for row in data:
        row["status"] = get_status(row)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        type_filter = st.selectbox(
            "Function Type",
            ["All", "classification", "regression"],
            index=0,
            key="overview_type_filter",
        )

    with col2:
        status_filter = st.selectbox(
            "Status",
            ["All", "Good", "Needs Attention", "Not Validated", "Missing"],
            index=0,
            key="overview_status_filter",
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Function Name", "R2 Score", "Samples", "Status"],
            index=0,
            key="overview_sort_by",
        )

    # Apply filters
    filtered = data
    if type_filter != "All":
        filtered = [d for d in filtered if d["function_type"] == type_filter]
    if status_filter != "All":
        filtered = [d for d in filtered if d["status"] == status_filter]

    # Sort
    sort_key_map = {
        "Function Name": lambda x: x["function_name"],
        "R2 Score": lambda x: x["latest_r2"] or 0,
        "Samples": lambda x: x["n_samples"] or 0,
        "Status": lambda x: ["Good", "Needs Attention", "Not Validated", "Missing"].index(
            x["status"]
        ),
    }
    filtered = sorted(
        filtered, key=sort_key_map[sort_by], reverse=(sort_by == "R2 Score" or sort_by == "Samples")
    )

    # Summary metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    status_counts = {}
    for d in data:
        status_counts[d["status"]] = status_counts.get(d["status"], 0) + 1

    with col1:
        st.metric("Good", status_counts.get("Good", 0))
    with col2:
        st.metric("Needs Attention", status_counts.get("Needs Attention", 0))
    with col3:
        st.metric("Not Validated", status_counts.get("Not Validated", 0))
    with col4:
        st.metric("Missing", status_counts.get("Missing", 0))

    st.divider()

    # Build dataframe for display
    df_data = []
    for row in filtered:
        df_data.append(
            {
                "Function": row["function_name"],
                "Type": row["function_type"],
                "Has Surrogate": "Yes" if row["has_surrogate"] else "No",
                "Samples": str(row["n_samples"]) if row["n_samples"] else "-",
                "Training R2": f"{row['training_r2']:.4f}" if row["training_r2"] else "-",
                "Validation R2": f"{row['latest_r2']:.4f}" if row["latest_r2"] else "-",
                "Status": row["status"],
            }
        )

    df = pd.DataFrame(df_data)

    # Style function for status column
    def highlight_status(val):
        color = get_status_color(val)
        return f"color: {color}; font-weight: bold"

    # Display table
    if len(df) > 0:
        styled_df = df.style.applymap(highlight_status, subset=["Status"])
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(df) + 38),
        )
    else:
        st.info("No functions match the selected filters.")

    # Export option
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "surrogate_overview.csv",
            "text/csv",
            use_container_width=True,
            key="overview_download_csv",
        )
