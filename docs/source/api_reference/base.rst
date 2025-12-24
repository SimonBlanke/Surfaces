.. _api_base:

==============
Base Classes
==============

Base classes that all test functions inherit from.

BaseTestFunction
================

The root class for all test functions in Surfaces.

.. autoclass:: surfaces.test_functions._base_test_function.BaseTestFunction
   :members:
   :undoc-members:
   :show-inheritance:

AlgebraicFunction
=================

Base class for algebraic (mathematical) test functions.

.. autoclass:: surfaces.test_functions.algebraic._base_algebraic_function.AlgebraicFunction
   :members:
   :show-inheritance:

EngineeringFunction
===================

Base class for engineering design optimization problems.

.. autoclass:: surfaces.test_functions.engineering._base_engineering_function.EngineeringFunction
   :members:
   :show-inheritance:
