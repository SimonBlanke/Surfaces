.. _api_bbob:

==============
BBOB Functions
==============

The 24 noiseless benchmark functions from the BBOB (Black-Box Optimization Benchmarking) test suite, part of the COCO platform.

Functions are organized into five categories based on their properties.

.. contents:: On this page
   :local:
   :depth: 2

----

Separable (f1-f5)
=================

Functions that can be optimized dimension-by-dimension.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.bbob.Sphere
   surfaces.test_functions.bbob.EllipsoidalSeparable
   surfaces.test_functions.bbob.RastriginSeparable
   surfaces.test_functions.bbob.BuecheRastrigin
   surfaces.test_functions.bbob.LinearSlope

----

Low/Moderate Conditioning (f6-f9)
=================================

Functions with condition numbers <= 10.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.bbob.AttractiveSector
   surfaces.test_functions.bbob.StepEllipsoidal
   surfaces.test_functions.bbob.RosenbrockOriginal
   surfaces.test_functions.bbob.RosenbrockRotated

----

High Conditioning & Unimodal (f10-f14)
======================================

Ill-conditioned unimodal functions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.bbob.EllipsoidalRotated
   surfaces.test_functions.bbob.Discus
   surfaces.test_functions.bbob.BentCigar
   surfaces.test_functions.bbob.SharpRidge
   surfaces.test_functions.bbob.DifferentPowers

----

Multimodal with Adequate Global Structure (f15-f19)
===================================================

Multimodal functions with a discernible global pattern.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.bbob.RastriginRotated
   surfaces.test_functions.bbob.Weierstrass
   surfaces.test_functions.bbob.SchaffersF7
   surfaces.test_functions.bbob.SchaffersF7Ill
   surfaces.test_functions.bbob.GriewankRosenbrock

----

Multimodal with Weak Global Structure (f20-f24)
===============================================

Highly deceptive multimodal functions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.bbob.Schwefel
   surfaces.test_functions.bbob.Gallagher101
   surfaces.test_functions.bbob.Gallagher21
   surfaces.test_functions.bbob.Katsuura
   surfaces.test_functions.bbob.LunacekBiRastrigin
