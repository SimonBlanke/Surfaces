.. _api_base:

========================
Base Classes & Interface
========================

.. include:: ../_generated/diagrams/base_classes_overview.rst

----

BaseTestFunction
================

The root class for all test functions in Surfaces.

.. autoclass:: surfaces.test_functions._base_test_function.BaseTestFunction
   :members:
   :undoc-members:
   :show-inheritance:

----

Algebraic Base Classes
======================

AlgebraicFunction
-----------------

Base class for algebraic (mathematical) test functions.

.. autoclass:: surfaces.test_functions.algebraic._base_algebraic_function.AlgebraicFunction
   :members:
   :show-inheritance:

BBOBFunction
^^^^^^^^^^^^

Base class for BBOB (Black-Box Optimization Benchmarking) functions.

.. autoclass:: surfaces.test_functions.benchmark.bbob._base_bbob.BBOBFunction
   :members:
   :show-inheritance:

CECFunction
^^^^^^^^^^^

Base class for CEC competition benchmark functions.

.. autoclass:: surfaces.test_functions.benchmark.cec._base_cec.CECFunction
   :members:
   :show-inheritance:

CEC2013Function
"""""""""""""""

.. autoclass:: surfaces.test_functions.benchmark.cec.cec2013._base_cec2013.CEC2013Function
   :members:
   :show-inheritance:

CEC2014Function
"""""""""""""""

.. autoclass:: surfaces.test_functions.benchmark.cec.cec2014._base_cec2014.CEC2014Function
   :members:
   :show-inheritance:

CEC2017Function
"""""""""""""""

.. autoclass:: surfaces.test_functions.benchmark.cec.cec2017._base_cec2017.CEC2017Function
   :members:
   :show-inheritance:

----

Machine Learning Base Classes
=============================

MachineLearningFunction
-----------------------

Base class for machine learning hyperparameter optimization functions.

.. autoclass:: surfaces.test_functions.machine_learning._base_machine_learning.MachineLearningFunction
   :members:
   :show-inheritance:

BaseTabular
^^^^^^^^^^^

Base class for tabular data ML functions.

.. autoclass:: surfaces.test_functions.machine_learning.tabular._base_tabular.BaseTabular
   :members:
   :show-inheritance:

BaseImage
^^^^^^^^^

Base class for image data ML functions.

.. autoclass:: surfaces.test_functions.machine_learning.image._base_image.BaseImage
   :members:
   :show-inheritance:

BaseTimeSeries
^^^^^^^^^^^^^^

Base class for time series ML functions.

.. autoclass:: surfaces.test_functions.machine_learning.timeseries._base_timeseries.BaseTimeSeries
   :members:
   :show-inheritance:

----

Engineering Base Classes
========================

EngineeringFunction
-------------------

Base class for engineering design optimization problems with constraints.

.. autoclass:: surfaces.test_functions.algebraic.constrained._base_constrained_function.EngineeringFunction
   :members:
   :show-inheritance:
