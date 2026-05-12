"""Tests for the `SemanticBundleOptimization._format_example_fields` helper.

These tests exercise the helper against the actual DSPy data structures it sees
at runtime — `Example.inputs()`, `Example.labels()`, and `Prediction` — across
the program signatures defined in `benchmarks/programs/qa.py` (single-field QA,
context-augmented QA, reasoning-first, math). They also cover the fallback
paths (non-mapping values) and verify that no silent truncation occurs.
"""

import pytest

from dspy import Example, Prediction
from dspy.teleprompt.sbo import SemanticBundleOptimization


@pytest.fixture
def sbo() -> SemanticBundleOptimization:
    """A minimally configured SBO instance — only `metric` is required."""
    return SemanticBundleOptimization(metric=lambda ex, pred, trace: 1.0)


# ---------------------------------------------------------------------------
# Example.inputs() / Example.labels() — the two main critique call sites.
# ---------------------------------------------------------------------------


def test_naive_qa_inputs_render_only_input_fields(sbo):
    """`question -> answer` signature: inputs() should render just 'question'."""
    ex = Example(
        question="What is the capital of France?",
        answer="Paris",
    ).with_inputs("question")

    rendered = sbo._format_example_fields(ex.inputs())

    assert rendered == "question: What is the capital of France?"
    assert "answer" not in rendered, "labels must not leak into inputs rendering"


def test_naive_qa_labels_render_only_label_fields(sbo):
    ex = Example(
        question="What is the capital of France?",
        answer="Paris",
    ).with_inputs("question")

    rendered = sbo._format_example_fields(ex.labels())

    assert rendered == "answer: Paris"
    assert "question" not in rendered, "inputs must not leak into labels rendering"


def test_context_qa_inputs_include_question_and_context(sbo):
    """`context, question -> answer` signature: both input fields must appear."""
    ex = Example(
        question="Who wrote Hamlet?",
        context="Hamlet is a tragedy written by William Shakespeare around 1600.",
        answer="William Shakespeare",
    ).with_inputs("question", "context")

    rendered = sbo._format_example_fields(ex.inputs())

    assert "question: Who wrote Hamlet?" in rendered
    assert "context: Hamlet is a tragedy" in rendered


def test_hotpotqa_style_labels_include_gold_titles(sbo):
    """HotPotQA examples carry `gold_titles` alongside `answer`; both are labels."""
    ex = Example(
        question="Which Bond actor is older?",
        answer="Sean Connery",
        gold_titles={"Sean Connery", "Roger Moore"},
    ).with_inputs("question")

    rendered = sbo._format_example_fields(ex.labels())

    assert "answer: Sean Connery" in rendered
    assert "gold_titles:" in rendered
    assert "question" not in rendered


def test_math_signature_uses_problem_not_question(sbo):
    """`problem -> answer` (MathNaive) must work without any hardcoded key list."""
    ex = Example(
        problem="What is 2 + 2?",
        answer="4",
    ).with_inputs("problem")

    inputs_rendered = sbo._format_example_fields(ex.inputs())
    labels_rendered = sbo._format_example_fields(ex.labels())

    assert inputs_rendered == "problem: What is 2 + 2?"
    assert labels_rendered == "answer: 4"


# ---------------------------------------------------------------------------
# Prediction — the third critique call site.
# ---------------------------------------------------------------------------


def test_prediction_renders_all_output_fields(sbo):
    """ReasoningFirstQA produces predictions with `reasoning` and `answer`."""
    pred = Prediction(
        reasoning="Paris has been France's capital since 987 AD.",
        answer="Paris",
    )

    rendered = sbo._format_example_fields(pred)

    assert "reasoning: Paris has been France's capital since 987 AD." in rendered
    assert "answer: Paris" in rendered


def test_prediction_with_single_field(sbo):
    """NaiveQA predictions have just one output field."""
    pred = Prediction(answer="Paris")

    rendered = sbo._format_example_fields(pred)

    assert rendered == "answer: Paris"


# ---------------------------------------------------------------------------
# Fallback paths — non-mapping values & broken inputs.
# ---------------------------------------------------------------------------


def test_plain_string_falls_back_to_str(sbo):
    assert sbo._format_example_fields("just a string") == "just a string"


def test_none_falls_back_to_str(sbo):
    assert sbo._format_example_fields(None) == "None"


def test_integer_falls_back_to_str(sbo):
    assert sbo._format_example_fields(42) == "42"


def test_object_with_broken_items_falls_back_to_str(sbo):
    """If `.items()` raises, the helper should still produce a string."""

    class BrokenItems:
        def items(self):
            raise RuntimeError("items() is broken")

        def __str__(self):
            return "<broken-object>"

    assert sbo._format_example_fields(BrokenItems()) == "<broken-object>"


def test_empty_example_renders_empty_string(sbo):
    """An Example with no remaining fields should produce an empty render."""
    ex = Example(question="hi").with_inputs("question")

    # `labels()` on an example whose only field is an input → no fields left.
    assert sbo._format_example_fields(ex.labels()) == ""


# ---------------------------------------------------------------------------
# No-truncation guarantees — the helper must preserve every field in full.
# ---------------------------------------------------------------------------


def test_short_text_passes_through_unchanged(sbo):
    ex = Example(question="short").with_inputs("question")

    rendered = sbo._format_example_fields(ex.inputs())

    assert rendered == "question: short"
    assert not rendered.endswith("...")


def test_long_single_field_is_not_truncated(sbo):
    """A long value (e.g., HotPotQA `context`) must be rendered in full."""
    long_context = "x" * 5000
    ex = Example(
        question="q",
        context=long_context,
    ).with_inputs("question", "context")

    rendered = sbo._format_example_fields(ex.inputs())

    assert long_context in rendered, "long field must not be clipped"
    assert not rendered.endswith("...")


def test_later_fields_are_preserved_after_long_earlier_field(sbo):
    """A long earlier field must never silently drop a later field (e.g., `answer`)."""
    ex = Example(
        context="x" * 5000,
        answer="this-answer-must-be-visible",
        gold_titles={"A", "B"},
    )  # no with_inputs → items() exposes everything in store order

    rendered = sbo._format_example_fields(ex)

    assert "answer: this-answer-must-be-visible" in rendered
    assert "gold_titles:" in rendered
    assert not rendered.endswith("...")


# ---------------------------------------------------------------------------
# End-to-end shape check: mirrors how the critique prompt assembles failures.
# ---------------------------------------------------------------------------


def test_critique_failure_block_shape(sbo):
    """Replicates the dict shape built inside `_generate_critique`."""
    ex = Example(
        question="What is the capital of France?",
        answer="Paris",
    ).with_inputs("question")
    pred = Prediction(answer="Lyon")

    failure = {
        "input": sbo._format_example_fields(ex.inputs()),
        "expected": sbo._format_example_fields(ex.labels()),
        "predicted": sbo._format_example_fields(pred),
        "score": 0.0,
    }

    assert failure["input"] == "question: What is the capital of France?"
    assert failure["expected"] == "answer: Paris"
    assert failure["predicted"] == "answer: Lyon"


if __name__ == "__main__":
    # Allow running as a script for a quick visual smoke check:
    #   python tests/teleprompt/test_sbo.py
    sbo_instance = SemanticBundleOptimization(metric=lambda *_: 1.0)

    demo_ex = Example(
        question="Who wrote Hamlet?",
        context="Hamlet was written by William Shakespeare.",
        answer="William Shakespeare",
        gold_titles={"Hamlet", "Shakespeare"},
    ).with_inputs("question", "context")
    demo_pred = Prediction(
        reasoning="Shakespeare authored Hamlet around 1600.",
        answer="William Shakespeare",
    )

    print("=== inputs() ===")
    print(sbo_instance._format_example_fields(demo_ex.inputs()))
    print("\n=== labels() ===")
    print(sbo_instance._format_example_fields(demo_ex.labels()))
    print("\n=== Prediction ===")
    print(sbo_instance._format_example_fields(demo_pred))
    print("\n=== Plain string ===")
    print(sbo_instance._format_example_fields("a bare value"))
    print("\n=== Long field (full passthrough) ===")
    long_demo = Example(question="q", context="x" * 200).with_inputs("question", "context")
    rendered_long = sbo_instance._format_example_fields(long_demo.inputs())
    print(f"length={len(rendered_long)}, ends_with_ellipsis={rendered_long.endswith('...')}")
