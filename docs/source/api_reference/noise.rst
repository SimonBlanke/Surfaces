.. _api_noise:

=====
Noise
=====

Noise layers for adding stochastic disturbances to test functions.

These classes can be passed to any test function to simulate noisy evaluations. Useful for testing algorithm robustness to measurement uncertainty.

.. contents:: On this page
   :local:
   :depth: 2

----

Base Class
==========

.. autoclass:: surfaces.noise.BaseNoise
   :members:
   :show-inheritance:

----

Noise Types
===========

GaussianNoise
-------------

Additive Gaussian noise: ``f(x) + N(0, sigma^2)``

.. autoclass:: surfaces.noise.GaussianNoise
   :members:
   :show-inheritance:

UniformNoise
------------

Additive uniform noise: ``f(x) + U(low, high)``

.. autoclass:: surfaces.noise.UniformNoise
   :members:
   :show-inheritance:

MultiplicativeNoise
-------------------

Multiplicative noise: ``f(x) * (1 + N(0, sigma^2))``

.. autoclass:: surfaces.noise.MultiplicativeNoise
   :members:
   :show-inheritance:
