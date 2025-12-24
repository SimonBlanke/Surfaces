# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Details page - Deep dive into a single function.
"""

import pandas as pd
import streamlit as st

from surfaces._surrogates._dashboard.database import (
    get_all_surrogates,
    get_surrogate,
    get_training_jobs,
    get_validation_runs,
)


def render():
    """Render the details page."""
    st.header("Function Details")

    # Get all functions
    surrogates = get_all_surrogates()
    all_names = [s["function_name"] for s in surrogates]

    if not all_names:
        st.warning("No functions found. Run sync to populate the database.")
        return

    # Function selector
    selected = st.selectbox(
        "Select Function",
        all_names,
        index=0,
        key="details_function_select",
    )

    if not selected:
        return

    # Get detailed info
    surrogate = get_surrogate(selected)

    if not surrogate:
        st.error(f"Function {selected} not found in database.")
        return

    st.divider()

    # Status banner
    if surrogate["has_surrogate"]:
        st.success(f"Surrogate model available for {selected}")
    else:
        st.warning(f"No surrogate model for {selected}")

    # Tabs for different info
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Metadata",
            "Parameters",
            "Training History",
            "Validation History",
        ]
    )

    with tab1:
        render_metadata(surrogate)

    with tab2:
        render_parameters(surrogate)

    with tab3:
        render_training_history(selected)

    with tab4:
        render_validation_history(selected)


def render_metadata(surrogate: dict):
    """Render metadata section."""
    st.subheader("Surrogate Metadata")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Info**")
        st.write(f"- Function Name: `{surrogate['function_name']}`")
        st.write(f"- Type: {surrogate['function_type']}")
        st.write(f"- Has Surrogate: {'Yes' if surrogate['has_surrogate'] else 'No'}")
        st.write(f"- Has Validity Model: {'Yes' if surrogate['has_validity_model'] else 'No'}")

    with col2:
        st.write("**Training Info**")
        if surrogate["has_surrogate"]:
            st.write(f"- Training Samples: {surrogate['n_samples'] or 'N/A'}")
            st.write(f"- Invalid Samples: {surrogate['n_invalid_samples'] or 0}")
            st.write(
                f"- Training R2: {surrogate['training_r2']:.4f}"
                if surrogate["training_r2"]
                else "- Training R2: N/A"
            )
            st.write(
                f"- Training MSE: {surrogate['training_mse']:.6f}"
                if surrogate["training_mse"]
                else "- Training MSE: N/A"
            )
            st.write(
                f"- Training Time: {surrogate['training_time_sec']:.1f}s"
                if surrogate["training_time_sec"]
                else "- Training Time: N/A"
            )
        else:
            st.write("No training data available.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Value Range**")
        if surrogate["y_range_min"] is not None:
            st.write(f"- Min: {surrogate['y_range_min']:.4f}")
            st.write(f"- Max: {surrogate['y_range_max']:.4f}")
        else:
            st.write("No range data available.")

    with col2:
        st.write("**Tracking Info**")
        st.write(
            f"- Last Synced: {surrogate['last_synced_at'][:19] if surrogate['last_synced_at'] else 'Never'}"
        )
        st.write(
            f"- Created: {surrogate['created_at'][:19] if surrogate['created_at'] else 'Unknown'}"
        )
        if surrogate["onnx_file_hash"]:
            st.write(f"- File Hash: `{surrogate['onnx_file_hash'][:16]}...`")


def render_parameters(surrogate: dict):
    """Render parameters section."""
    st.subheader("Model Parameters")

    param_names = surrogate.get("param_names", [])
    param_encodings = surrogate.get("param_encodings", {})

    if not param_names:
        st.info("No parameter information available.")
        return

    st.write(f"**Parameter Count:** {len(param_names)}")

    # Parameter table
    param_data = []
    for name in param_names:
        encoding = param_encodings.get(name)
        if encoding:
            param_type = "categorical"
            values = list(encoding.keys())
            values_str = ", ".join(values[:5])
            if len(values) > 5:
                values_str += f" (+{len(values) - 5} more)"
        else:
            param_type = "numeric"
            values_str = "-"

        param_data.append(
            {
                "Parameter": name,
                "Type": param_type,
                "Values": values_str,
            }
        )

    df = pd.DataFrame(param_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Raw encodings
    if param_encodings:
        with st.expander("View Raw Encodings"):
            st.json(param_encodings)


def render_training_history(function_name: str):
    """Render training history section."""
    st.subheader("Training History")

    jobs = get_training_jobs(function_name=function_name, limit=20)

    if not jobs:
        st.info("No training history for this function.")
        return

    df_data = []
    for job in jobs:
        duration = "-"
        if job["started_at"] and job["completed_at"]:
            # Simple duration calculation
            try:
                from datetime import datetime

                start = datetime.fromisoformat(job["started_at"])
                end = datetime.fromisoformat(job["completed_at"])
                dur_sec = (end - start).total_seconds()
                duration = f"{dur_sec:.1f}s"
            except Exception:
                pass

        df_data.append(
            {
                "Started": job["started_at"][:19] if job["started_at"] else "-",
                "Duration": duration,
                "Status": job["status"],
                "Triggered By": job["triggered_by"],
                "Error": job["error_message"][:30] + "..."
                if job["error_message"] and len(job["error_message"]) > 30
                else (job["error_message"] or "-"),
            }
        )

    df = pd.DataFrame(df_data)

    def highlight_status(val):
        if val == "completed":
            return "color: #28a745"
        elif val == "failed":
            return "color: #dc3545"
        elif val == "running":
            return "color: #17a2b8"
        return ""

    styled_df = df.style.applymap(highlight_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_validation_history(function_name: str):
    """Render validation history section."""
    st.subheader("Validation History")

    runs = get_validation_runs(function_name=function_name, limit=20)

    if not runs:
        st.info("No validation history for this function.")
        return

    # Summary chart
    if len(runs) > 1:
        st.write("**R2 Score Trend**")

        # Prepare data for chart
        chart_data = []
        for run in reversed(runs):  # Oldest first for chart
            if run["r2_score"] is not None:
                chart_data.append(
                    {
                        "Date": run["validated_at"][:10] if run["validated_at"] else "",
                        "R2": run["r2_score"],
                    }
                )

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            st.line_chart(chart_df.set_index("Date"))

    # Table
    st.write("**Validation Runs**")

    df_data = []
    for run in runs:
        df_data.append(
            {
                "Date": run["validated_at"][:19] if run["validated_at"] else "-",
                "Type": run["validation_type"],
                "Samples": str(run["n_samples"]) if run["n_samples"] else "-",
                "R2": f"{run['r2_score']:.4f}" if run["r2_score"] else "-",
                "MAE": f"{run['mae']:.4f}" if run["mae"] else "-",
                "RMSE": f"{run['rmse']:.4f}" if run["rmse"] else "-",
                "Max Error": f"{run['max_error']:.4f}" if run["max_error"] else "-",
                "Speedup": f"{run['speedup_factor']:.0f}x" if run["speedup_factor"] else "-",
            }
        )

    df = pd.DataFrame(df_data)

    def highlight_r2(val):
        try:
            r2 = float(val)
            if r2 >= 0.95:
                return "color: #28a745"
            elif r2 >= 0.90:
                return "color: #ffc107"
            else:
                return "color: #dc3545"
        except (ValueError, TypeError):
            return ""

    styled_df = df.style.applymap(highlight_r2, subset=["R2"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
