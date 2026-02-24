"""Discovery utilities for ML function test classes.

Provides functions to automatically discover all concrete ML function
classes in the surfaces package via recursive module scanning.

Used by the test_ml_*.py test modules in this package.
"""

import importlib
import inspect
import pkgutil
from typing import List, Type

from surfaces.test_functions.machine_learning._base_machine_learning import (
    MachineLearningFunction,
)

# Base/abstract class names excluded from concrete class discovery
_BASE_CLASS_NAMES = {
    "MachineLearningFunction",
    "BaseTabular",
    "BaseClassification",
    "BaseRegression",
    "BaseForecaster",
    "BaseTSClassifier",
    "BaseImageClassification",
    "BaseNeuralArchitectureSearch",
    "BaseNASFunction",
    "BaseTransferLearning",
    "BaseDataAugmentation",
    "BaseReinforcementLearning",
    "BaseFeatureEngineering",
    "BaseEnsembleOptimization",
    "BasePipeline",
    "BaseLLMOptimization",
    "BaseTabularFeatureEngineering",
    "BaseTabularEnsemble",
    "BaseTabularPipeline",
    "BaseTimeSeries",
    "BaseImage",
}


def _is_concrete_ml_class(cls: type) -> bool:
    """Check if a class is a concrete ML function (not base/abstract)."""
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, MachineLearningFunction):
        return False
    if cls is MachineLearningFunction:
        return False
    if cls.__name__ in _BASE_CLASS_NAMES:
        return False
    if cls.__name__.startswith("Base"):
        return False
    return True


def discover_ml_classes() -> List[Type[MachineLearningFunction]]:
    """Discover all concrete ML function classes via recursive module scan.

    Uses pkgutil.walk_packages to traverse all modules under
    surfaces.test_functions.machine_learning, then inspects each for
    classes inheriting from MachineLearningFunction.

    This discovers classes regardless of whether they appear in __init__.py
    exports, making it suitable for catching unregistered implementations.
    """
    import surfaces.test_functions.machine_learning as ml_root

    discovered: set = set()

    def _scan(module) -> None:
        for name in dir(module):
            obj = getattr(module, name)
            if _is_concrete_ml_class(obj):
                discovered.add(obj)

        if hasattr(module, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(
                module.__path__, prefix=module.__name__ + "."
            ):
                if "._" in modname:
                    continue
                try:
                    submod = importlib.import_module(modname)
                    _scan(submod)
                except (ImportError, Exception):
                    # Module may require unavailable optional deps
                    pass

    _scan(ml_root)
    return sorted(discovered, key=lambda x: x.__name__)


def ml_class_id(cls: type) -> str:
    """Generate a readable test ID from an ML class."""
    return cls.__name__


# Pre-computed at import time for use in @pytest.mark.parametrize
ML_CLASSES = discover_ml_classes()
