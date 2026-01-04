# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2017 Simple Benchmark Functions (F1-F10)."""

from typing import Any, Dict

import numpy as np

from ._base_cec2017 import CEC2017Function


class ShiftedRotatedBentCigar(CEC2017Function):
    """F1: Shifted and Rotated Bent Cigar Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Bent Cigar Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    # Function sheet attributes
    latex_formula = r"f(\vec{z}) = z_1^2 + 10^6 \sum_{i=2}^{n} z_i^2 \quad \text{where } \vec{z} = M(\vec{x} - \vec{o})"
    tagline = (
        "A shifted and rotated ill-conditioned function. "
        "One dimension dominates, creating a narrow valley in transformed space."
    )
    display_bounds = (-100.0, 100.0)
    display_projection = {"fixed_value": 0.0}
    reference = "CEC 2017 Competition"
    reference_url = "https://github.com/P-N-Suganthan/CEC2017-BoundConstrained"

    def _create_objective_function(self) -> None:
        def bent_cigar(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + self.f_global

        self.pure_objective_function = bent_cigar


class ShiftedRotatedSumDiffPow(CEC2017Function):
    """F2: Shifted and Rotated Sum of Different Power Function (DEPRECATED).

    Note: This function has been deprecated from the CEC 2017 benchmark suite.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Sum of Different Power Function",
        "func_id": 2,
        "unimodal": True,
        "separable": False,
        "deprecated": True,
    }

    def _create_objective_function(self) -> None:
        def sum_diff_pow(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            D = self.n_dim
            result = sum(abs(z[i]) ** (i + 1) for i in range(D))
            return result + self.f_global

        self.pure_objective_function = sum_diff_pow


class ShiftedRotatedZakharov(CEC2017Function):
    """F3: Shifted and Rotated Zakharov Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Zakharov Function",
        "func_id": 3,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def zakharov(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            D = self.n_dim
            sum1 = np.sum(z**2)
            sum2 = np.sum(0.5 * np.arange(1, D + 1) * z)
            return sum1 + sum2**2 + sum2**4 + self.f_global

        self.pure_objective_function = zakharov


class ShiftedRotatedRosenbrock(CEC2017Function):
    """F4: Shifted and Rotated Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Rosenbrock's Function",
        "func_id": 4,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rosenbrock(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 0.02048 * z + 1.0

            result = 0.0
            for i in range(self.n_dim - 1):
                result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2

            return result + self.f_global

        self.pure_objective_function = rosenbrock


class ShiftedRotatedRastrigin(CEC2017Function):
    """F5: Shifted and Rotated Rastrigin's Function.

    Properties:
    - Highly multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Rastrigin's Function",
        "func_id": 5,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 0.0512 * z

            D = self.n_dim
            result = 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

            return result + self.f_global

        self.pure_objective_function = rastrigin


class ShiftedRotatedSchafferF7(CEC2017Function):
    """F6: Shifted and Rotated Schaffer's F7 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Schaffer's F7 Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schaffers_f7(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            D = self.n_dim
            si = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
            tmp = np.sin(50 * (si**0.2))
            sm = np.sum(np.sqrt(si) * (tmp**2 + 1))
            result = (sm**2) / ((D - 1) ** 2)

            return result + self.f_global

        self.pure_objective_function = schaffers_f7


class ShiftedRotatedLunacekBiRastrigin(CEC2017Function):
    """F7: Shifted and Rotated Lunacek Bi-Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Two global optima
    """

    _spec = {
        "name": "Shifted and Rotated Lunacek Bi-Rastrigin's Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def lunacek_bi_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            shift = self._get_shift_vector()
            M = self._get_rotation_matrix()

            D = self.n_dim
            mu0 = 2.5
            s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
            mu1 = -np.sqrt((mu0**2 - 1) / s)

            y = 0.1 * (x - shift)
            tmpx = 2 * y.copy()
            tmpx[shift < 0] *= -1
            z = tmpx.copy()
            tmpx = tmpx + mu0

            t1 = np.sum((tmpx - mu0) ** 2)
            t2 = s * np.sum((tmpx - mu1) ** 2) + D

            y = M @ z
            t = np.sum(np.cos(2 * np.pi * y))

            result = min(t1, t2) + 10 * (D - t)

            return result + self.f_global

        self.pure_objective_function = lunacek_bi_rastrigin


class ShiftedRotatedNonContRastrigin(CEC2017Function):
    """F8: Shifted and Rotated Non-Continuous Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-continuous
    """

    _spec = {
        "name": "Shifted and Rotated Non-Continuous Rastrigin's Function",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
        "continuous": False,
    }

    def _create_objective_function(self) -> None:
        def non_cont_rastrigin(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            shift = self._get_shift_vector()
            M = self._get_rotation_matrix()

            shifted = x - shift
            x_mod = x.copy()
            mask = np.abs(shifted) > 0.5
            x_mod[mask] = (shift + np.floor(2 * shifted + 0.5) * 0.5)[mask]

            z = 0.0512 * shifted
            z = M @ z

            D = self.n_dim
            result = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

            return result + self.f_global

        self.pure_objective_function = non_cont_rastrigin


class ShiftedRotatedLevy(CEC2017Function):
    """F9: Shifted and Rotated Levy Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    _spec = {
        "name": "Shifted and Rotated Levy Function",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def levy(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)

            w = 1.0 + 0.25 * (z - 1.0)
            term1 = np.sin(np.pi * w[0]) ** 2
            term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
            sm = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))

            return term1 + sm + term3 + self.f_global

        self.pure_objective_function = levy


class ShiftedRotatedSchwefel(CEC2017Function):
    """F10: Shifted and Rotated Schwefel's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Deceptive
    """

    _spec = {
        "name": "Shifted and Rotated Schwefel's Function",
        "func_id": 10,
        "unimodal": False,
        "separable": False,
    }

    def _create_objective_function(self) -> None:
        def schwefel(params: Dict[str, Any]) -> float:
            x = self._params_to_array(params)
            z = self._shift_rotate(x)
            z = 10.0 * z + 420.9687462275036

            D = self.n_dim
            result = 0.0
            for i in range(D):
                zi = z[i]
                if abs(zi) <= 500:
                    result += zi * np.sin(np.sqrt(abs(zi)))
                elif zi > 500:
                    zm = 500 - zi % 500
                    result += zm * np.sin(np.sqrt(abs(zm)))
                    result -= (zi - 500) ** 2 / (10000 * D)
                else:
                    zm = abs(zi) % 500 - 500
                    result += zm * np.sin(np.sqrt(abs(zm)))
                    result -= (zi + 500) ** 2 / (10000 * D)

            return 418.9829 * D - result + self.f_global

        self.pure_objective_function = schwefel
