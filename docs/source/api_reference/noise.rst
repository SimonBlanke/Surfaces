.. _api_noise:

=====
Noise
=====

.. include:: ../_generated/diagrams/noise_overview.rst

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
