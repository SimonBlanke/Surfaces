# Machine Learning Test Functions - Parameter Coverage

This document shows the parameter ranges covered by the database for each machine learning test function in the Surfaces library.

The database stores evaluation results for specific parameter combinations to enable fast lookup of expensive ML computations. Each function has default parameter ranges defined in code, and this document shows how much of that space is covered by stored data.

## Legend
- ✅ **Complete**: All default parameter combinations are stored
- 🟡 **Good coverage**: 80-99% of default combinations stored  
- 🟠 **Partial coverage**: 50-79% of default combinations stored
- 🔴 **Limited coverage**: <50% of default combinations stored
- ❌ **Not found**: Parameter not found in stored data

## Gradient Boosting Regressor Function

**Function ID:** `gradient_boosting_regressor`

ℹ️ **Status:** Database exists but contains no data

## KNeighbors Regressor Function

**Function ID:** `k_neighbors_regressor`

**Total evaluations stored:** 1

### Parameter Coverage

| Parameter | Type | Stored Values | Coverage Range | Default Range | Coverage |
|-----------|------|---------------|----------------|---------------|----------|
| n_neighbors | numeric | 1 | 3.0 - 3.0 | N/A | 🔴 3.3% |
| algorithm | categorical | 1 | ['auto'] | 4 categories | 🔴 25.0% |
| cv | numeric | 1 | 2.0 - 2.0 | 2 - 10 | 🔴 16.7% |
| dataset | numeric | 1 | ['diabetes_data'] | N/A | ✅ Complete |

## KNeighbors Classifier Function

**Function ID:** `k_neighbors_classifier`

**Total evaluations stored:** 2

### Parameter Coverage

| Parameter | Type | Stored Values | Coverage Range | Default Range | Coverage |
|-----------|------|---------------|----------------|---------------|----------|
| n_neighbors | numeric | 2 | 3.0 - 200.0 | N/A | 🔴 6.7% |
| algorithm | categorical | 1 | ['auto'] | 4 categories | 🔴 25.0% |
| cv | numeric | 1 | 2.0 - 2.0 | 2 - 10 | 🔴 16.7% |
| dataset | numeric | 1 | ['iris_data'] | N/A | 🔴 33.3% |


---

---

*Generated on 2025-08-09 14:19:51*
