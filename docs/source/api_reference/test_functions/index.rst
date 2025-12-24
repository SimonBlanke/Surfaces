.. _api_test_functions:

==============
Test Functions
==============

All available test functions in Surfaces, organized by category.

.. contents:: On this page
   :local:
   :depth: 2

----

Algebraic Functions
===================

Classic mathematical optimization test functions from the literature.

1D Functions
------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.algebraic.test_functions_1d.gramacy_and_lee_function.GramacyAndLeeFunction

2D Functions
------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.algebraic.test_functions_2d.ackley_function.AckleyFunction
   surfaces.test_functions.algebraic.test_functions_2d.beale_function.BealeFunction
   surfaces.test_functions.algebraic.test_functions_2d.booth_function.BoothFunction
   surfaces.test_functions.algebraic.test_functions_2d.bukin_function_n6.BukinFunctionN6
   surfaces.test_functions.algebraic.test_functions_2d.cross_in_tray_function.CrossInTrayFunction
   surfaces.test_functions.algebraic.test_functions_2d.drop_wave_function.DropWaveFunction
   surfaces.test_functions.algebraic.test_functions_2d.easom_function.EasomFunction
   surfaces.test_functions.algebraic.test_functions_2d.eggholder_function.EggholderFunction
   surfaces.test_functions.algebraic.test_functions_2d.goldstein_price_function.GoldsteinPriceFunction
   surfaces.test_functions.algebraic.test_functions_2d.himmelblaus_function.HimmelblausFunction
   surfaces.test_functions.algebraic.test_functions_2d.hoelder_table_function.HölderTableFunction
   surfaces.test_functions.algebraic.test_functions_2d.langermann_function.LangermannFunction
   surfaces.test_functions.algebraic.test_functions_2d.levi_function_n13.LeviFunctionN13
   surfaces.test_functions.algebraic.test_functions_2d.matyas_function.MatyasFunction
   surfaces.test_functions.algebraic.test_functions_2d.mccormick_function.McCormickFunction
   surfaces.test_functions.algebraic.test_functions_2d.schaffer_function_n2.SchafferFunctionN2
   surfaces.test_functions.algebraic.test_functions_2d.simionescu_function.SimionescuFunction
   surfaces.test_functions.algebraic.test_functions_2d.three_hump_camel_function.ThreeHumpCamelFunction

N-D Functions
-------------

Scalable functions that work with any number of dimensions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.algebraic.test_functions_nd.griewank_function.GriewankFunction
   surfaces.test_functions.algebraic.test_functions_nd.rastrigin_function.RastriginFunction
   surfaces.test_functions.algebraic.test_functions_nd.rosenbrock_function.RosenbrockFunction
   surfaces.test_functions.algebraic.test_functions_nd.sphere_function.SphereFunction
   surfaces.test_functions.algebraic.test_functions_nd.styblinski_tang_function.StyblinskiTangFunction

----

Machine Learning Functions
==========================

Hyperparameter optimization landscapes for machine learning models.

Tabular Classification
----------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.tabular.classification.test_functions.decision_tree_classifier.DecisionTreeClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.gradient_boosting_classifier.GradientBoostingClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.k_neighbors_classifier.KNeighborsClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.random_forest_classifier.RandomForestClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.svm_classifier.SVMClassifierFunction

Tabular Regression
------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.tabular.regression.test_functions.decision_tree_regressor.DecisionTreeRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor.GradientBoostingRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.k_neighbors_regressor.KNeighborsRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.random_forest_regressor.RandomForestRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.svm_regressor.SVMRegressorFunction

Image Classification
--------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.image.classification.test_functions.simple_cnn_classifier.SimpleCNNClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.deep_cnn_classifier.DeepCNNClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.svm_image_classifier.SVMImageClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.random_forest_image_classifier.RandomForestImageClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.xgboost_image_classifier.XGBoostImageClassifierFunction

Time Series Classification
--------------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.random_forest_ts_classifier.RandomForestTSClassifierFunction
   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.knn_ts_classifier.KNNTSClassifierFunction
   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.ts_forest_classifier.TSForestClassifierFunction

Time Series Forecasting
-----------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.gradient_boosting_forecaster.GradientBoostingForecasterFunction
   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.random_forest_forecaster.RandomForestForecasterFunction
   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.exp_smoothing_forecaster.ExpSmoothingForecasterFunction

----

Engineering Functions
=====================

Real-world constrained engineering design optimization problems.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.engineering.pressure_vessel.PressureVesselFunction
   surfaces.test_functions.engineering.tension_compression_spring.TensionCompressionSpringFunction
   surfaces.test_functions.engineering.three_bar_truss.ThreeBarTrussFunction
   surfaces.test_functions.engineering.welded_beam.WeldedBeamFunction
   surfaces.test_functions.engineering.cantilever_beam.CantileverBeamFunction
