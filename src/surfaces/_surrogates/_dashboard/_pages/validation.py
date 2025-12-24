# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Validation page - Validate surrogate accuracy.
"""

import numpy as np
import pandas as pd
import streamlit as st

from surfaces._surrogates._dashboard.database import get_all_surrogates, get_validation_runs
from surfaces._surrogates._dashboard.validation import validate_all, validate_surrogate


def render():
    """Render the validation page."""
    st.header("Validation")

    # Get current state
    surrogates = get_all_surrogates()
    with_surrogate = [s for s in surrogates if s["has_surrogate"]]
    names_with_surrogate = [s["function_name"] for s in with_surrogate]

    if not with_surrogate:
        st.warning("No surrogates available to validate. Train some surrogates first.")
        return

    # Validation settings
    st.subheader("Validation Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        validation_type = st.selectbox(
            "Validation Type",
            ["random", "grid"],
            index=0,
            key="validation_type_select",
        )

    with col2:
        n_samples = st.number_input(
            "Number of Samples",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key="validation_n_samples",
        )

    with col3:
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            key="validation_random_seed",
        )

    # Batch validation
    st.divider()
    st.subheader("Batch Validation")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button(
            f"Validate All ({len(with_surrogate)})",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.validate_all = True

    # Handle batch validation
    if st.session_state.get("validate_all", False):
        st.divider()
        st.subheader("Validation Progress")

        logs = []
        log_container = st.container()

        def progress_callback(msg: str):
            logs.append(msg)
            with log_container:
                st.text("\n".join(logs[-20:]))

        with st.spinner("Validating all surrogates..."):
            results = validate_all(validation_type, n_samples, random_seed, progress_callback)

        success_count = sum(1 for r in results if r["success"])
        fail_count = len(results) - success_count

        if fail_count == 0:
            st.success(f"Successfully validated {success_count} surrogate(s)")
        else:
            st.warning(f"Validated {success_count}, Failed {fail_count}")

        del st.session_state.validate_all
        st.rerun()

    # Single function validation
    st.divider()
    st.subheader("Validate Single Function")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_function = st.selectbox(
            "Select Function",
            names_with_surrogate,
            index=0 if names_with_surrogate else None,
            label_visibility="collapsed",
            key="validate_single_select",
        )

    with col2:
        validate_single = st.button(
            "Validate", use_container_width=True, type="primary", key="validation_validate_btn"
        )

    if validate_single and selected_function:
        st.divider()
        st.subheader("Validation Progress")

        logs = []
        log_container = st.container()

        def progress_callback(msg: str):
            logs.append(msg)
            with log_container:
                st.text("\n".join(logs))

        with st.spinner(f"Validating {selected_function}..."):
            result = validate_surrogate(
                selected_function,
                validation_type,
                n_samples,
                random_seed,
                progress_callback,
            )

        if result["success"]:
            st.success(result["message"])

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = result["metrics"]
            timing = result["timing"]

            with col1:
                st.metric("R2 Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col4:
                st.metric("Speedup", f"{timing['speedup']:.0f}x")

            # Scatter plot
            if result["data"] is not None:
                st.subheader("Predicted vs Actual")

                data = result["data"]
                y_real = data["y_real"]
                y_surr = data["y_surrogate"]

                # Create scatter plot data
                scatter_df = pd.DataFrame(
                    {
                        "Actual": y_real,
                        "Predicted": y_surr,
                    }
                )

                # Plot
                try:
                    import plotly.express as px

                    fig = px.scatter(
                        scatter_df,
                        x="Actual",
                        y="Predicted",
                        title="Surrogate vs Real Function",
                    )

                    # Add diagonal line
                    min_val = min(y_real.min(), y_surr.min())
                    max_val = max(y_real.max(), y_surr.max())
                    fig.add_shape(
                        type="line",
                        x0=min_val,
                        y0=min_val,
                        x1=max_val,
                        y1=max_val,
                        line=dict(color="red", dash="dash"),
                    )

                    fig.update_layout(
                        xaxis_title="Actual Value",
                        yaxis_title="Predicted Value",
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    # Fallback to simple display
                    st.scatter_chart(scatter_df, x="Actual", y="Predicted")

                # Error distribution
                st.subheader("Error Distribution")
                errors = data["errors"]

                error_df = pd.DataFrame({"Error": errors})

                try:
                    import plotly.express as px

                    fig = px.histogram(
                        error_df,
                        x="Error",
                        nbins=30,
                        title="Prediction Error Distribution",
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.bar_chart(np.histogram(errors, bins=30)[0])

        else:
            st.error(result["message"])

    # Validation history
    st.divider()
    st.subheader("Validation History")

    runs = get_validation_runs(limit=50)

    if runs:
        df_data = []
        for run in runs:
            df_data.append(
                {
                    "Function": run["function_name"],
                    "Type": run["validation_type"],
                    "Samples": str(run["n_samples"]) if run["n_samples"] else "-",
                    "R2": f"{run['r2_score']:.4f}" if run["r2_score"] else "-",
                    "MAE": f"{run['mae']:.4f}" if run["mae"] else "-",
                    "Speedup": f"{run['speedup_factor']:.0f}x" if run["speedup_factor"] else "-",
                    "Date": run["validated_at"][:19] if run["validated_at"] else "-",
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
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(df) + 38),
        )
    else:
        st.info("No validation runs recorded yet.")
