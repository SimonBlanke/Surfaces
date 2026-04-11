# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .._base_image import BaseImage


class BaseImageClassification(BaseImage):
    """Base class for image classification test functions."""

    def _subsample_data(self, X, y, fidelity: float):
        """Stratified subsampling to preserve class distribution."""
        rng = np.random.RandomState(42)
        classes = np.unique(y)
        indices = []
        for c in classes:
            c_idx = np.where(y == c)[0]
            n_c = max(1, int(len(c_idx) * fidelity))
            chosen = rng.choice(c_idx, size=n_c, replace=False)
            indices.append(chosen)
        indices = np.sort(np.concatenate(indices))
        return X[indices], y[indices]
