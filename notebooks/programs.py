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


class MLflowBasePromptContextQA(dspy.Module):
    """QA module with a structured base prompt for context-based question answering.
    
    This module implements the base prompt from the MLflow blog post on systematic 
    prompt optimization with GEPA:
    https://mlflow.org/blog/mlflow-prompt-optimization
    
    The prompt is designed to:
    - Answer questions based ONLY on the provided context
    - For yes/no questions, answer ONLY "yes" or "no"
    - NOT include phrases like "based on the context" or "according to the documents"
    
    Supports HotPotQA dict format, list of strings, or a single string.
    """

    def __init__(self):
        super().__init__()
        # Create a signature with a detailed instruction as the docstring
        class ContextQASignature(dspy.Signature):
            """You are a question answering assistant. Answer questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- For yes/no questions, answer ONLY "yes" or "no"
- Do NOT include phrases like "based on the context" or "according to the documents"
"""
            context: str = dspy.InputField(desc="The context to use for answering the question")
            question: str = dspy.InputField(desc="The question to answer")
            answer: str = dspy.OutputField(desc="The answer to the question")
        
        self.answer = dspy.Predict(ContextQASignature)

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
    "mlflow_base_prompt": MLflowBasePromptContextQA,
}


def create_program(program_type: str = "naive") -> dspy.Module:
    """Create and return a fresh QA program instance.

    Args:
        program_type: Type of program to create. Options:
            - "naive": Direct question-to-answer prediction
            - "reasoning": Two-stage reasoning-first approach
            - "context": Uses provided context to answer questions
            - "reasoning_context": Two-stage reasoning with context
            - "mlflow_base_prompt": MLflow blog base prompt for HotPotQA (https://mlflow.org/blog/mlflow-prompt-optimization)

    Returns:
        A fresh instance of the specified program type.

    Raises:
        ValueError: If program_type is not recognized.
    """
    if program_type not in PROGRAMS:
        available = ", ".join(PROGRAMS.keys())
        raise ValueError(f"Unknown program type: {program_type}. Available: {available}")
    return PROGRAMS[program_type]()
