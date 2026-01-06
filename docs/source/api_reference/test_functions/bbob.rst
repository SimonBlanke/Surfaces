.. _api_bbob:

==============
BBOB Functions
==============

.. include:: ../../_generated/diagrams/bbob_overview.rst

----

Separable (f1-f5)
=================

Functions that can be optimized dimension-by-dimension.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.benchmark.bbob.Sphere
   surfaces.test_functions.benchmark.bbob.EllipsoidalSeparable
   surfaces.test_functions.benchmark.bbob.RastriginSeparable
   surfaces.test_functions.benchmark.bbob.BuecheRastrigin
   surfaces.test_functions.benchmark.bbob.LinearSlope

----

Low/Moderate Conditioning (f6-f9)
=================================

Functions with condition numbers <= 10.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.benchmark.bbob.AttractiveSector
   surfaces.test_functions.benchmark.bbob.StepEllipsoidal
   surfaces.test_functions.benchmark.bbob.RosenbrockOriginal
   surfaces.test_functions.benchmark.bbob.RosenbrockRotated

----

High Conditioning & Unimodal (f10-f14)
======================================

Ill-conditioned unimodal functions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.benchmark.bbob.EllipsoidalRotated
   surfaces.test_functions.benchmark.bbob.Discus
   surfaces.test_functions.benchmark.bbob.BentCigar
   surfaces.test_functions.benchmark.bbob.SharpRidge
   surfaces.test_functions.benchmark.bbob.DifferentPowers

----

Multimodal with Adequate Global Structure (f15-f19)
===================================================

Multimodal functions with a discernible global pattern.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.benchmark.bbob.RastriginRotated
   surfaces.test_functions.benchmark.bbob.Weierstrass
   surfaces.test_functions.benchmark.bbob.SchaffersF7
   surfaces.test_functions.benchmark.bbob.SchaffersF7Ill
   surfaces.test_functions.benchmark.bbob.GriewankRosenbrock

----

Multimodal with Weak Global Structure (f20-f24)
===============================================

Highly deceptive multimodal functions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.benchmark.bbob.Schwefel
   surfaces.test_functions.benchmark.bbob.Gallagher101
   surfaces.test_functions.benchmark.bbob.Gallagher21
   surfaces.test_functions.benchmark.bbob.Katsuura
   surfaces.test_functions.benchmark.bbob.LunacekBiRastrigin
