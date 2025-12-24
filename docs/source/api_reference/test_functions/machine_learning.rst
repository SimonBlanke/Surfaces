.. _api_machine_learning:

==========================
Machine Learning Functions
==========================

Hyperparameter optimization landscapes for machine learning models.

.. contents:: On this page
   :local:
   :depth: 2

----

Tabular Data
============

Functions for optimizing ML models on tabular datasets.

Classification
--------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.tabular.classification.test_functions.decision_tree_classifier.DecisionTreeClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.gradient_boosting_classifier.GradientBoostingClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.k_neighbors_classifier.KNeighborsClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.random_forest_classifier.RandomForestClassifierFunction
   surfaces.test_functions.machine_learning.tabular.classification.test_functions.svm_classifier.SVMClassifierFunction

Regression
----------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.tabular.regression.test_functions.decision_tree_regressor.DecisionTreeRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor.GradientBoostingRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.k_neighbors_regressor.KNeighborsRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.random_forest_regressor.RandomForestRegressorFunction
   surfaces.test_functions.machine_learning.tabular.regression.test_functions.svm_regressor.SVMRegressorFunction

----

Image Data
==========

Functions for optimizing ML models on image datasets.

Classification
--------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.image.classification.test_functions.simple_cnn_classifier.SimpleCNNClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.deep_cnn_classifier.DeepCNNClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.svm_image_classifier.SVMImageClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.random_forest_image_classifier.RandomForestImageClassifierFunction
   surfaces.test_functions.machine_learning.image.classification.test_functions.xgboost_image_classifier.XGBoostImageClassifierFunction

----

Time Series
===========

Functions for optimizing ML models on time series data.

Classification
--------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.random_forest_ts_classifier.RandomForestTSClassifierFunction
   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.knn_ts_classifier.KNNTSClassifierFunction
   surfaces.test_functions.machine_learning.timeseries.classification.test_functions.ts_forest_classifier.TSForestClassifierFunction

Forecasting
-----------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst
   :nosignatures:

   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.gradient_boosting_forecaster.GradientBoostingForecasterFunction
   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.random_forest_forecaster.RandomForestForecasterFunction
   surfaces.test_functions.machine_learning.timeseries.forecasting.test_functions.exp_smoothing_forecaster.ExpSmoothingForecasterFunction
