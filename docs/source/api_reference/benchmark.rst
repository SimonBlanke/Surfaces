.. _api_benchmark:

=========
Benchmark
=========

The ``surfaces.benchmark`` module for systematic optimizer comparison.
See the :ref:`user_guide_benchmark` for usage examples and concepts.

----

Core
====

Benchmark
---------

.. autoclass:: surfaces.benchmark.Benchmark
   :members:
   :show-inheritance:

Suite
-----

.. autoclass:: surfaces.benchmark.Suite
   :members:
   :show-inheritance:

----

Result Analysis
===============

ResultAccessor
--------------

Accessed via ``bench.results``.

.. autoclass:: surfaces.benchmark._accessors.ResultAccessor
   :members:
   :show-inheritance:

FriedmanResult
--------------

Returned by ``bench.results.friedman()``.

.. autoclass:: surfaces.benchmark._statistics.FriedmanResult
   :members:
   :show-inheritance:

ERTTable
--------

Returned by ``bench.results.ert()``. Subscriptable by function name,
then optimizer name.

.. autoclass:: surfaces.benchmark._statistics.ERTTable
   :members:
   :show-inheritance:

ERTEntry
--------

.. autoclass:: surfaces.benchmark._statistics.ERTEntry
   :members:
   :show-inheritance:

RankingTable
------------

Returned by ``bench.results.ranking()``. Subscriptable by optimizer name.

.. autoclass:: surfaces.benchmark._statistics.RankingTable
   :members:
   :show-inheritance:

RankingEntry
------------

.. autoclass:: surfaces.benchmark._statistics.RankingEntry
   :members:
   :show-inheritance:

----

Persistence
===========

IOAccessor
----------

Accessed via ``bench.io``.

.. autoclass:: surfaces.benchmark._accessors.IOAccessor
   :members:
   :show-inheritance:

----

Visualization
=============

PlotAccessor
------------

Accessed via ``bench.plot``.

.. autoclass:: surfaces.benchmark._accessors.PlotAccessor
   :members:
   :show-inheritance:

----

Traces
======

Trace
-----

One complete optimization trajectory (single optimizer, single function,
single seed).

.. autoclass:: surfaces.benchmark.Trace
   :members:
   :show-inheritance:

EvalRecord
----------

.. autoclass:: surfaces.benchmark.EvalRecord
   :members:
   :show-inheritance:

----

Execution
=========

TrialInfo
---------

Passed to callbacks after each trial completes.

.. autoclass:: surfaces.benchmark.TrialInfo
   :members:
   :show-inheritance:

ParallelBackend
---------------

.. autoclass:: surfaces.benchmark.ParallelBackend
   :members:
   :show-inheritance:

ProcessBackend
--------------

.. autoclass:: surfaces.benchmark.ProcessBackend
   :members:
   :show-inheritance:

ThreadBackend
-------------

.. autoclass:: surfaces.benchmark.ThreadBackend
   :members:
   :show-inheritance:

----

Statistical Functions
=====================

These are called internally by the accessor methods but can also be used
standalone.

.. autofunction:: surfaces.benchmark._statistics.compute_friedman

.. autofunction:: surfaces.benchmark._statistics.compute_ert

.. autofunction:: surfaces.benchmark._statistics.compute_ranking
