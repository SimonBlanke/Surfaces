# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model loader for fast function evaluation.

Loads pre-trained ONNX models and provides a callable interface
that matches the pure_objective_function signature.
"""

from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
import json

import numpy as np


class SurrogateLoader:
    """Load and run ONNX surrogate models.

    Parameters
    ----------
    model_path : Path or str
        Path to the .onnx model file.
    metadata_path : Path or str, optional
        Path to the metadata JSON file. If None, looks for
        {model_path}.meta.json

    Attributes
    ----------
    param_names : list
        Ordered parameter names expected by the model.
    param_encodings : dict
        Mappings for categorical parameters to numeric values.
    """

    def __init__(
        self,
        model_path: Path,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = metadata_path or self.model_path.with_suffix(
            self.model_path.suffix + ".meta.json"
        )

        self._session = None
        self._metadata = None

    @property
    def session(self):
        """Lazy-load ONNX runtime session."""
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError(
                    "onnxruntime is required for surrogate models. "
                    "Install it with: pip install onnxruntime"
                )

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._session

    @property
    def metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if self._metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {self.metadata_path}. "
                    "Surrogate models require a metadata file."
                )
            with open(self.metadata_path, "r") as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def param_names(self) -> List[str]:
        """Ordered parameter names expected by the model."""
        return self.metadata["param_names"]

    @property
    def param_encodings(self) -> Dict[str, Dict[str, int]]:
        """Mappings for categorical parameters."""
        return self.metadata.get("param_encodings", {})

    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array for model input.

        Handles:
        - Numeric parameters: pass through
        - Categorical parameters: encode using param_encodings
        - Callable parameters (datasets): encode using string representation
        """
        values = []
        for name in self.param_names:
            value = params[name]

            if name in self.param_encodings:
                encoding = self.param_encodings[name]
                # Handle callable (like dataset functions)
                if callable(value):
                    key = value.__name__
                else:
                    key = str(value)
                value = encoding.get(key, 0)

            values.append(float(value))

        return np.array([values], dtype=np.float32)

    def predict(self, params: Dict[str, Any]) -> float:
        """Run inference on the surrogate model.

        Parameters
        ----------
        params : dict
            Parameter dictionary matching pure_objective_function signature.

        Returns
        -------
        float
            Predicted objective value.
        """
        input_array = self._encode_params(params)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_array})
        return float(output[0][0, 0])

    def as_objective_function(self) -> Callable[[Dict[str, Any]], float]:
        """Return a callable matching pure_objective_function signature.

        Returns
        -------
        callable
            Function that takes a params dict and returns a float.
        """
        return self.predict


def get_surrogate_path(function_name: str) -> Optional[Path]:
    """Get the path to a pre-trained surrogate model.

    Parameters
    ----------
    function_name : str
        Name of the function (e.g., "k_neighbors_classifier").

    Returns
    -------
    Path or None
        Path to the ONNX file if it exists, None otherwise.
    """
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / f"{function_name}.onnx"

    if model_path.exists():
        return model_path
    return None


def load_surrogate(function_name: str) -> Optional[SurrogateLoader]:
    """Load a pre-trained surrogate model by function name.

    Parameters
    ----------
    function_name : str
        Name of the function (e.g., "k_neighbors_classifier").

    Returns
    -------
    SurrogateLoader or None
        Loaded surrogate if available, None otherwise.
    """
    model_path = get_surrogate_path(function_name)
    if model_path is None:
        return None
    return SurrogateLoader(model_path)
