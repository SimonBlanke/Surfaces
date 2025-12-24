.. _api_cec:

=============
CEC Functions
=============

.. include:: ../../_generated/diagrams/cec_overview.rst

----

CEC 2013
========

28 functions: unimodal, multimodal, and composition.

Unimodal (F1-F5)
----------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2013.Sphere
   surfaces.test_functions.cec.cec2013.RotatedHighConditionedElliptic
   surfaces.test_functions.cec.cec2013.RotatedBentCigar
   surfaces.test_functions.cec.cec2013.RotatedDiscus
   surfaces.test_functions.cec.cec2013.DifferentPowers

Multimodal (F6-F20)
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2013.RotatedRosenbrock
   surfaces.test_functions.cec.cec2013.RotatedSchafferF7
   surfaces.test_functions.cec.cec2013.RotatedAckley
   surfaces.test_functions.cec.cec2013.RotatedWeierstrass
   surfaces.test_functions.cec.cec2013.RotatedGriewank
   surfaces.test_functions.cec.cec2013.Rastrigin
   surfaces.test_functions.cec.cec2013.RotatedRastrigin
   surfaces.test_functions.cec.cec2013.StepRastrigin
   surfaces.test_functions.cec.cec2013.Schwefel
   surfaces.test_functions.cec.cec2013.RotatedSchwefel
   surfaces.test_functions.cec.cec2013.RotatedKatsuura
   surfaces.test_functions.cec.cec2013.LunacekBiRastrigin
   surfaces.test_functions.cec.cec2013.RotatedLunacekBiRastrigin
   surfaces.test_functions.cec.cec2013.RotatedExpandedGriewankRosenbrock
   surfaces.test_functions.cec.cec2013.RotatedExpandedScafferF6

Composition (F21-F28)
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2013.CompositionFunction1
   surfaces.test_functions.cec.cec2013.CompositionFunction2
   surfaces.test_functions.cec.cec2013.CompositionFunction3
   surfaces.test_functions.cec.cec2013.CompositionFunction4
   surfaces.test_functions.cec.cec2013.CompositionFunction5
   surfaces.test_functions.cec.cec2013.CompositionFunction6
   surfaces.test_functions.cec.cec2013.CompositionFunction7
   surfaces.test_functions.cec.cec2013.CompositionFunction8

----

CEC 2014
========

30 functions: unimodal, multimodal, hybrid, and composition.

Unimodal (F1-F3)
----------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2014.RotatedHighConditionedElliptic
   surfaces.test_functions.cec.cec2014.RotatedBentCigar
   surfaces.test_functions.cec.cec2014.RotatedDiscus

Multimodal (F4-F16)
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2014.ShiftedRotatedRosenbrock
   surfaces.test_functions.cec.cec2014.ShiftedRotatedAckley
   surfaces.test_functions.cec.cec2014.ShiftedRotatedWeierstrass
   surfaces.test_functions.cec.cec2014.ShiftedRotatedGriewank
   surfaces.test_functions.cec.cec2014.ShiftedRastrigin
   surfaces.test_functions.cec.cec2014.ShiftedRotatedRastrigin
   surfaces.test_functions.cec.cec2014.ShiftedSchwefel
   surfaces.test_functions.cec.cec2014.ShiftedRotatedSchwefel
   surfaces.test_functions.cec.cec2014.ShiftedRotatedKatsuura
   surfaces.test_functions.cec.cec2014.ShiftedRotatedHappyCat
   surfaces.test_functions.cec.cec2014.ShiftedRotatedHGBat
   surfaces.test_functions.cec.cec2014.ShiftedRotatedExpandedGriewankRosenbrock
   surfaces.test_functions.cec.cec2014.ShiftedRotatedExpandedScafferF6

Hybrid (F17-F22)
----------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2014.HybridFunction1
   surfaces.test_functions.cec.cec2014.HybridFunction2
   surfaces.test_functions.cec.cec2014.HybridFunction3
   surfaces.test_functions.cec.cec2014.HybridFunction4
   surfaces.test_functions.cec.cec2014.HybridFunction5
   surfaces.test_functions.cec.cec2014.HybridFunction6

Composition (F23-F30)
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2014.CompositionFunction1
   surfaces.test_functions.cec.cec2014.CompositionFunction2
   surfaces.test_functions.cec.cec2014.CompositionFunction3
   surfaces.test_functions.cec.cec2014.CompositionFunction4
   surfaces.test_functions.cec.cec2014.CompositionFunction5
   surfaces.test_functions.cec.cec2014.CompositionFunction6
   surfaces.test_functions.cec.cec2014.CompositionFunction7
   surfaces.test_functions.cec.cec2014.CompositionFunction8

----

CEC 2017
========

30 functions: simple, hybrid, and composition. Note: F2 is deprecated.

Simple (F1-F10)
---------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.cec.cec2017.ShiftedRotatedBentCigar
   surfaces.test_functions.cec.cec2017.ShiftedRotatedSumDiffPow
   surfaces.test_functions.cec.cec2017.ShiftedRotatedZakharov
   surfaces.test_functions.cec.cec2017.ShiftedRotatedRosenbrock
   surfaces.test_functions.cec.cec2017.ShiftedRotatedRastrigin
   surfaces.test_functions.cec.cec2017.ShiftedRotatedSchafferF7
   surfaces.test_functions.cec.cec2017.ShiftedRotatedLunacekBiRastrigin
   surfaces.test_functions.cec.cec2017.ShiftedRotatedNonContRastrigin
   surfaces.test_functions.cec.cec2017.ShiftedRotatedLevy
   surfaces.test_functions.cec.cec2017.ShiftedRotatedSchwefel
