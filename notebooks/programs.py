"""
DSPy Program Definitions

This module contains reusable DSPy program/module definitions for various tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from dspy import Prediction

class ReasoningFirstQA(dspy.Module):
    """Multi-step QA module with separate think and answer predictors.

    This module uses a two-stage approach:
    1. Think: Analyze the question and generate reasoning
    2. Answer: Use the reasoning to produce a concise answer

    GEPA and other optimizers can optimize both predictor instructions independently.
    """

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought(
            dspy.Signature(
                "question -> reasoning",
            )
        )
        self.answer = dspy.Predict(
            dspy.Signature(
                "question, reasoning -> answer",
            )
        )

    def forward(self, question: str) -> Prediction:
        think_result = self.think(question=question)
        return self.answer(question=question, reasoning=think_result.reasoning)
    
class NaiveQA(dspy.Module):
    """Naive QA module that directly predicts the answer from the question.
    """

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "question -> answer",
            )
        )

    def forward(self, question: str) -> Prediction:
        return self.answer(question=question)


# Registry of available programs
PROGRAMS = {
    "naive": NaiveQA,
    "reasoning": ReasoningFirstQA,
}


def create_program(program_type: str = "naive") -> dspy.Module:
    """Create and return a fresh QA program instance.

    Args:
        program_type: Type of program to create. Options:
            - "naive": Direct question-to-answer prediction
            - "reasoning": Two-stage reasoning-first approach

    Returns:
        A fresh instance of the specified program type.

    Raises:
        ValueError: If program_type is not recognized.
    """
    if program_type not in PROGRAMS:
        available = ", ".join(PROGRAMS.keys())
        raise ValueError(f"Unknown program type: {program_type}. Available: {available}")
    return PROGRAMS[program_type]()
