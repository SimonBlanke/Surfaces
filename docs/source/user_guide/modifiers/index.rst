.. _user_guide_modifiers:

=========
Modifiers
=========

Modifiers transform test functions to create new optimization challenges.
They allow you to test how your optimizer handles various real-world
complications.

----

Available Modifiers
===================

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item-card:: Noise
      :link: noise
      :link-type: doc

      Add stochastic noise to function evaluations.
      Test optimizer robustness to noisy objectives.

----

Why Use Modifiers?
==================

Real-world optimization problems are rarely clean:

- **Noisy evaluations**: Measurements have variance
- **Shifted optima**: The optimum may not be at zero
- **Rotated spaces**: Variables may be correlated
- **Scaled outputs**: Different magnitude ranges

Modifiers let you test these scenarios systematically.

----

Using Modifiers
===============

Modifiers wrap existing test functions:

.. code-block:: python

    from surfaces.test_functions import SphereFunction
    from surfaces.modifiers import GaussianNoise

    # Create function with noise modifier
    noisy_func = SphereFunction(
        n_dim=5,
        modifiers=[GaussianNoise(sigma=0.1)]
    )

    # Evaluate - now includes random noise
    result = noisy_func({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0})

    # Get true value without noise
    true_result = noisy_func.true_value({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0})

----

Future Modifiers
================

Planned modifiers for future releases:

- **Shift**: Move the global optimum to a new location
- **Rotation**: Apply rotation matrix to input space
- **Scaling**: Scale input or output values
- **Composition**: Combine multiple functions
- **Discretization**: Convert continuous to discrete parameters

----

.. toctree::
   :maxdepth: 1
   :hidden:

   noise
