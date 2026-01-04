#!/usr/bin/env python3
"""Demonstrate surrogate model usage for fast ML function evaluation.

This example shows how the refactored KNeighborsClassifierFunction works
with dataset/cv as init parameters (like coefficients) and surrogates
for fast evaluation.
"""

import time

from surfaces.test_functions import KNeighborsClassifierFunction


def main():
    print("=" * 60)
    print("Surrogate Model Demo")
    print("=" * 60)

    # 1. Basic usage - dataset and cv are init parameters
    print("\n1. Basic API (dataset/cv as init params)")
    print("-" * 40)
    func = KNeighborsClassifierFunction(dataset="iris", cv=5)
    print(f"   dataset: {func.dataset}")
    print(f"   cv: {func.cv}")
    print(f"   search_space: {list(func.search_space.keys())}")
    print("   -> Only hyperparameters in search space!")

    # 2. Real evaluation (slow)
    print("\n2. Real Evaluation")
    print("-" * 40)
    func_real = KNeighborsClassifierFunction(dataset="digits", cv=5, use_surrogate=False)

    start = time.time()
    for i in range(10):
        result = func_real({"n_neighbors": 5 + i * 10, "algorithm": "auto"})
    elapsed = time.time() - start

    print(f"   10 evaluations: {elapsed*1000:.1f}ms ({elapsed/10*1000:.1f}ms/eval)")

    # 3. Surrogate evaluation (fast)
    print("\n3. Surrogate Evaluation")
    print("-" * 40)
    func_surr = KNeighborsClassifierFunction(dataset="digits", cv=5, use_surrogate=True)

    # Warm-up (first call loads model)
    _ = func_surr({"n_neighbors": 5, "algorithm": "auto"})

    start = time.time()
    for i in range(1000):
        result = func_surr({"n_neighbors": 5 + (i % 30) * 5, "algorithm": "auto"})
    elapsed = time.time() - start

    print(f"   1000 evaluations: {elapsed*1000:.1f}ms ({elapsed/1000*1000:.3f}ms/eval)")

    # 4. Compare accuracy
    print("\n4. Accuracy Comparison")
    print("-" * 40)
    test_params = [
        {"n_neighbors": 3, "algorithm": "auto"},
        {"n_neighbors": 15, "algorithm": "ball_tree"},
        {"n_neighbors": 50, "algorithm": "kd_tree"},
    ]

    func_real = KNeighborsClassifierFunction(dataset="iris", cv=5, use_surrogate=False)
    func_surr = KNeighborsClassifierFunction(dataset="iris", cv=5, use_surrogate=True)

    print("   n_neighbors | Real   | Surrogate | Diff")
    print("   " + "-" * 40)
    for params in test_params:
        real = func_real(params)
        surr = func_surr(params)
        diff = abs(real - surr)
        print(f"   {params['n_neighbors']:11} | {real:.4f} | {surr:.4f}    | {diff:.4f}")

    # 5. Different datasets
    print("\n5. Works Across Datasets")
    print("-" * 40)
    for dataset in ["digits", "iris", "wine"]:
        func = KNeighborsClassifierFunction(dataset=dataset, cv=5, use_surrogate=True)
        result = func({"n_neighbors": 10, "algorithm": "auto"})
        print(f"   {dataset:8}: {result:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
