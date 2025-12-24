# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Training page - Train and retrain surrogates.
"""

import pandas as pd
import streamlit as st

from surfaces._surrogates._dashboard.database import get_all_surrogates, get_training_jobs
from surfaces._surrogates._dashboard.sync import sync_all
from surfaces._surrogates._dashboard.training import (
    train_all,
    train_low_accuracy,
    train_missing,
    train_surrogate,
)

# R2 threshold
R2_THRESHOLD = 0.95


def render():
    """Render the training page."""
    st.header("Training")

    # Get current state
    surrogates = get_all_surrogates()
    missing = [s for s in surrogates if not s["has_surrogate"]]
    all_names = [s["function_name"] for s in surrogates]

    # Batch actions
    st.subheader("Batch Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_count = len(missing)
        st.write(f"**Missing Surrogates:** {missing_count}")
        if st.button(
            f"Train Missing ({missing_count})",
            disabled=missing_count == 0,
            use_container_width=True,
            type="primary" if missing_count > 0 else "secondary",
        ):
            st.session_state.training_action = "missing"

    with col2:
        st.write(f"**R2 Threshold:** {R2_THRESHOLD}")
        if st.button(
            "Retrain Low Accuracy",
            use_container_width=True,
        ):
            st.session_state.training_action = "low_accuracy"

    with col3:
        st.write(f"**Total Functions:** {len(all_names)}")
        if st.button(
            "Retrain All",
            use_container_width=True,
        ):
            st.session_state.training_action = "all"

    # Handle batch actions
    if "training_action" in st.session_state:
        action = st.session_state.training_action

        st.divider()
        st.subheader("Training Progress")

        progress_container = st.empty()
        log_container = st.container()

        logs = []

        def progress_callback(msg: str):
            logs.append(msg)
            with log_container:
                st.text("\n".join(logs[-20:]))  # Show last 20 lines

        with st.spinner(f"Running {action} training..."):
            if action == "missing":
                results = train_missing(progress_callback)
            elif action == "low_accuracy":
                results = train_low_accuracy(R2_THRESHOLD, progress_callback)
            elif action == "all":
                results = train_all(progress_callback)
            else:
                results = []

        # Show results
        if results:
            success_count = sum(1 for r in results if r["success"])
            fail_count = len(results) - success_count

            if fail_count == 0:
                st.success(f"Successfully trained {success_count} surrogate(s)")
            else:
                st.warning(f"Trained {success_count}, Failed {fail_count}")

            # Re-sync database
            sync_all()
            st.rerun()

        del st.session_state.training_action

    # Single function training
    st.divider()
    st.subheader("Train Single Function")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_function = st.selectbox(
            "Select Function",
            all_names,
            index=0 if all_names else None,
            label_visibility="collapsed",
            key="training_function_select",
        )

    with col2:
        train_single = st.button(
            "Train", use_container_width=True, type="primary", key="training_train_btn"
        )

    if train_single and selected_function:
        st.divider()
        st.subheader("Training Progress")

        logs = []
        log_container = st.container()

        def progress_callback(msg: str):
            logs.append(msg)
            with log_container:
                st.text("\n".join(logs))

        with st.spinner(f"Training {selected_function}..."):
            result = train_surrogate(selected_function, "manual", progress_callback)

        if result["success"]:
            st.success(result["message"])
            if result["metrics"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R2 Score", f"{result['metrics']['r2']:.4f}")
                with col2:
                    st.metric("Samples", result["metrics"]["n_samples"])
                with col3:
                    st.metric("Training Time", f"{result['metrics']['training_time']:.1f}s")
            sync_all()
        else:
            st.error(result["message"])

    # Training history
    st.divider()
    st.subheader("Training History")

    jobs = get_training_jobs(limit=50)

    if jobs:
        df_data = []
        for job in jobs:
            df_data.append(
                {
                    "Function": job["function_name"],
                    "Started": job["started_at"][:19] if job["started_at"] else "-",
                    "Completed": job["completed_at"][:19] if job["completed_at"] else "-",
                    "Status": job["status"],
                    "Triggered By": job["triggered_by"],
                    "Error": job["error_message"][:50] + "..."
                    if job["error_message"] and len(job["error_message"]) > 50
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
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(df) + 38),
        )
    else:
        st.info("No training jobs recorded yet.")
