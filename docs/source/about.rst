.. _about:

=====
About
=====

Learn more about the Surfaces project.

Project Overview
================

Surfaces is a Python library that provides single-objective black-box
optimization test functions for benchmarking. It's part of a larger
ecosystem of optimization tools developed by Simon Blanke.

History
=======

Surfaces was created to provide a standardized set of test functions for
evaluating optimization algorithms, particularly those in the
`Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
and `Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_ projects.

The library draws from the classic optimization literature, implementing
well-known test functions like Ackley, Rosenbrock, and Rastrigin, while
also introducing novel ML-based test functions that provide realistic
hyperparameter optimization landscapes.

Related Projects
================

Surfaces is part of an ecosystem of optimization tools:

Gradient-Free-Optimizers
------------------------

`Gradient-Free-Optimizers <https://github.com/SimonBlanke/Gradient-Free-Optimizers>`_
provides implementations of gradient-free optimization algorithms, from
simple hill climbing to Bayesian optimization.

Hyperactive
-----------

`Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_ is a high-level
hyperparameter optimization framework that uses Gradient-Free-Optimizers
under the hood and provides integrations with scikit-learn, PyTorch, and more.

Author
======

Surfaces is developed and maintained by **Simon Blanke**.

- GitHub: `@SimonBlanke <https://github.com/SimonBlanke>`_
- Email: simon.blanke@yahoo.com

License
=======

Surfaces is released under the MIT License, which allows for both personal
and commercial use, modification, and distribution.

.. code-block:: text

    MIT License

    Copyright (c) 2020 - |current_year| Simon Blanke

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

Acknowledgments
===============

Thanks to all contributors and users of Surfaces who have helped improve
the library through bug reports, feature requests, and code contributions.

The mathematical test functions are based on the extensive optimization
literature. Key references include:

- Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing.
- Rosenbrock, H. H. (1960). An Automatic Method for Finding the Greatest
  or Least Value of a Function.
- Rastrigin, L. A. (1974). Systems of Extremal Control.

.. toctree::
   :maxdepth: 1
   :hidden:

   about/history
   about/license
   about/team
