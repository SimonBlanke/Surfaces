.. _api_engineering:

=====================
Engineering Functions
=====================

Real-world constrained engineering design optimization problems.

These functions represent actual engineering design scenarios with physical constraints. They are useful for testing constrained optimization algorithms.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.engineering.pressure_vessel.PressureVesselFunction
   surfaces.test_functions.engineering.tension_compression_spring.TensionCompressionSpringFunction
   surfaces.test_functions.engineering.three_bar_truss.ThreeBarTrussFunction
   surfaces.test_functions.engineering.welded_beam.WeldedBeamFunction
   surfaces.test_functions.engineering.cantilever_beam.CantileverBeamFunction
