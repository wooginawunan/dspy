"""
DSPy Program Definitions

This module contains reusable DSPy program/module definitions for various tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from dspy import Prediction


def format_context(context: dict | list[str] | str | None) -> str:
    """Format context into a string suitable for prompting.
    
    Handles multiple formats:
    - HotPotQA dict format: {'title': [...], 'sentences': [[...], ...]}
    - List of strings: ["passage 1", "passage 2", ...]
    - Single string: "passage text"
    """
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n\n".join(context)
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        # HotPotQA format
        passages = []
        for title, sentences in zip(context["title"], context["sentences"]):
            text = " ".join(sentences)
            passages.append(f"[{title}]\n{text}")
        return "\n\n".join(passages)
    return str(context)

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


class ContextQA(dspy.Module):
    """QA module that uses provided context to answer questions.
    
    This module takes both a question and context (e.g., retrieved passages)
    and uses them together to generate an answer.
    
    Supports HotPotQA dict format, list of strings, or a single string.
    """

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "context, question -> answer",
            )
        )

    def forward(self, question: str, context: dict | list[str] | str = None) -> Prediction:
        context_str = format_context(context)
        return self.answer(context=context_str, question=question)


class ReasoningContextQA(dspy.Module):
    """Multi-step QA module that uses context with reasoning.

    This module uses a two-stage approach with context:
    1. Think: Analyze the context and question to generate reasoning
    2. Answer: Use the reasoning to produce a concise answer
    
    Supports HotPotQA dict format, list of strings, or a single string.
    """

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought(
            dspy.Signature(
                "context, question -> reasoning",
            )
        )
        self.answer = dspy.Predict(
            dspy.Signature(
                "context, question, reasoning -> answer",
            )
        )

    def forward(self, question: str, context: dict | list[str] | str = None) -> Prediction:
        context_str = format_context(context)
        think_result = self.think(context=context_str, question=question)
        return self.answer(context=context_str, question=question, reasoning=think_result.reasoning)

    
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
    "context": ContextQA,
    "reasoning_context": ReasoningContextQA,
}


def create_program(program_type: str = "naive") -> dspy.Module:
    """Create and return a fresh QA program instance.

    Args:
        program_type: Type of program to create. Options:
            - "naive": Direct question-to-answer prediction
            - "reasoning": Two-stage reasoning-first approach
            - "context": Uses provided context to answer questions
            - "reasoning_context": Two-stage reasoning with context

    Returns:
        A fresh instance of the specified program type.

    Raises:
        ValueError: If program_type is not recognized.
    """
    if program_type not in PROGRAMS:
        available = ", ".join(PROGRAMS.keys())
        raise ValueError(f"Unknown program type: {program_type}. Available: {available}")
    return PROGRAMS[program_type]()
