"""Prompt Engineering Optimization (Mock Mode)."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_llm_optimization import BaseLLMOptimization


class PromptEngineeringFunction(BaseLLMOptimization):
    """Prompt Engineering test function for LLM optimization (Mock Mode).

    **What is optimized:**
    This function optimizes prompt structure and LLM sampling parameters for
    improved task performance. The search includes:
    - Instruction style (direct, polite, step_by_step, expert_role)
    - Number of few-shot examples (0, 1, 3, 5)
    - Chain-of-thought prompting (True/False)
    - Temperature for sampling (0.0, 0.3, 0.5, 0.7, 1.0)
    - Top-p nucleus sampling (0.8, 0.9, 0.95, 1.0)

    **Prompt engineering concept:**
    Prompt engineering is the art and science of designing inputs to LLMs to
    elicit desired outputs. Different instruction styles, examples, and sampling
    parameters can dramatically affect output quality, consistency, and creativity.

    **What the score means (MOCK MODE):**
    The score (0.0 to 1.0) represents a *heuristic estimate* of prompt quality
    based on known best practices from prompt engineering research:
    - Step-by-step instructions improve reasoning tasks
    - Few-shot examples help with pattern recognition (diminishing returns)
    - Chain-of-thought improves complex reasoning
    - Lower temperature improves consistency for factual tasks
    - Random noise simulates real-world variability

    **IMPORTANT - Mock Mode:**
    This function uses a HEURISTIC scoring function, NOT real LLM API calls.
    This makes it fast and free for prototyping optimization algorithms. To use
    real LLM evaluation, implement a custom objective function that calls your
    preferred LLM API (OpenAI, Anthropic, etc.) and evaluates on actual tasks.

    **Optimization goal:**
    MAXIMIZE the heuristic quality score. In real applications, you would maximize
    accuracy on a specific task (classification, summarization, question-answering).
    The goal is to find the combination of prompt structure and sampling parameters
    that produces the best task performance.

    **Computational cost:**
    Very low in mock mode (~0.001s per evaluation). Real LLM evaluation would be
    expensive (1-10s per evaluation depending on API and task).

    Parameters
    ----------
    task_type : str, default="reasoning"
        Type of task being optimized. One of: "reasoning", "creative", "factual".
        Affects the heuristic scoring (e.g., reasoning benefits from chain-of-thought).
    noise_level : float, default=0.05
        Standard deviation of Gaussian noise added to scores to simulate real-world
        variability in LLM outputs.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning import PromptEngineeringFunction
    >>> func = PromptEngineeringFunction(task_type="reasoning")
    >>> func.search_space
    {'instruction_style': ['direct', 'polite', 'step_by_step', ...], ...}
    >>> result = func({"instruction_style": "step_by_step", "num_examples": 3,
    ...                "chain_of_thought": True, "temperature": 0.3, "top_p": 0.9})
    >>> print(f"Heuristic quality score: {result:.4f}")

    Notes
    -----
    This is a PROTOTYPE function demonstrating prompt optimization without requiring
    LLM API access. For production use:
    1. Implement a real evaluation function that calls your LLM API
    2. Evaluate prompts on a labeled test set for your specific task
    3. Return actual task performance metrics (accuracy, F1, ROUGE, etc.)
    """

    name = "Prompt Engineering (Mock)"
    _name_ = "prompt_engineering"
    __name__ = "PromptEngineeringFunction"

    para_names = ["instruction_style", "num_examples", "chain_of_thought", "temperature", "top_p"]
    instruction_style_default = ["direct", "polite", "step_by_step", "expert_role"]
    num_examples_default = [0, 1, 3, 5]
    chain_of_thought_default = [True, False]
    temperature_default = [0.0, 0.3, 0.5, 0.7, 1.0]
    top_p_default = [0.8, 0.9, 0.95, 1.0]

    def __init__(
        self,
        task_type: str = "reasoning",
        noise_level: float = 0.05,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        if task_type not in ["reasoning", "creative", "factual"]:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Available: reasoning, creative, factual"
            )

        self.task_type = task_type
        self.noise_level = noise_level

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for prompt engineering optimization."""
        return {
            "instruction_style": self.instruction_style_default,
            "num_examples": self.num_examples_default,
            "chain_of_thought": self.chain_of_thought_default,
            "temperature": self.temperature_default,
            "top_p": self.top_p_default,
        }

    def _create_objective_function(self) -> None:
        """Create heuristic objective function for prompt engineering."""
        task_type = self.task_type
        noise_level = self.noise_level

        def objective_function(params: Dict[str, Any]) -> float:
            # Base score
            score = 0.5

            # Instruction style impact
            style_scores = {
                "reasoning": {
                    "step_by_step": 0.15,
                    "expert_role": 0.10,
                    "polite": 0.05,
                    "direct": 0.02,
                },
                "creative": {
                    "expert_role": 0.12,
                    "polite": 0.10,
                    "direct": 0.08,
                    "step_by_step": 0.05,
                },
                "factual": {
                    "direct": 0.15,
                    "expert_role": 0.12,
                    "step_by_step": 0.08,
                    "polite": 0.05,
                },
            }
            score += style_scores[task_type][params["instruction_style"]]

            # Few-shot examples (diminishing returns)
            score += min(0.2, params["num_examples"] * 0.05)

            # Chain-of-thought benefit (task-dependent)
            if params["chain_of_thought"]:
                cot_benefit = {"reasoning": 0.15, "creative": 0.05, "factual": 0.08}
                score += cot_benefit[task_type]

            # Temperature impact (task-dependent)
            temp = params["temperature"]
            if task_type == "factual":
                # Lower temperature better for factual tasks
                score += 0.1 * (1 - temp)
            elif task_type == "creative":
                # Medium temperature better for creative tasks
                score += 0.1 * (1 - abs(temp - 0.7))
            else:  # reasoning
                # Low-medium temperature for reasoning
                score += 0.1 * (1 - abs(temp - 0.3))

            # Top-p impact (slight preference for lower values for consistency)
            score += 0.05 * (1.0 - params["top_p"])

            # Add noise to simulate real-world variability
            score += np.random.normal(0, noise_level)

            # Clip to [0, 1]
            score = np.clip(score, 0.0, 1.0)

            return score

        self.pure_objective_function = objective_function
