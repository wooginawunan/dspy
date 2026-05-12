"""Tests for `SemanticBundleOptimization` internals.

Covers:
- `_format_example_fields` against real DSPy data structures (`Example.inputs()`,
  `Example.labels()`, `Prediction`) across the program signatures used in
  `benchmarks/programs/qa.py`, plus fallback paths and no-truncation guarantees.
- `_judge_single` (backed by `dspy.Predict(JudgeSemanticAlignment)`):
  clamping, default-on-failure, and structured-output coercion.
- `_generate_candidates` (backed by `dspy.Predict(ProposeCandidates)`):
  building candidate programs, padding, and graceful proposer failure.
- `_generate_critique` and `_generate_failure_critique` (backed by
  `dspy.Predict(DiagnoseWeakness)` / `dspy.Predict(DiagnoseFailedCandidate)`):
  failure sampling, formatting, and module-level fallbacks.
"""

import pytest

import dspy
from dspy import Example, Prediction
from dspy.teleprompt.sbo import (
    DiagnoseFailedCandidate,
    DiagnoseWeakness,
    JudgeSemanticAlignment,
    ProposeCandidates,
    SemanticBundleOptimization,
)
from dspy.utils.dummies import DummyLM


class _SingleSlotProgram(dspy.Module):
    """Minimal program with one predictor — used as the proposer center."""

    def __init__(self, instruction: str = "Answer the question."):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature("question -> answer", instruction)
        )

    def forward(self, question: str):
        return self.answer(question=question)


class _DivergentMultiSlotProgram(dspy.Module):
    """Two-predictor program whose predictors carry different task instructions."""

    def __init__(self):
        super().__init__()
        self.think = dspy.Predict(
            dspy.Signature("question -> reasoning", "Think step by step.")
        )
        self.answer = dspy.Predict(
            dspy.Signature("question, reasoning -> answer", "Output only the final answer.")
        )

    def forward(self, question: str):
        reasoning = self.think(question=question).reasoning
        return self.answer(question=question, reasoning=reasoning)


class _UnifiedMultiSlotProgram(dspy.Module):
    """Two-predictor program whose predictors share the same task instruction."""

    SHARED = "Answer carefully and concisely."

    def __init__(self):
        super().__init__()
        self.think = dspy.Predict(dspy.Signature("question -> reasoning", self.SHARED))
        self.answer = dspy.Predict(dspy.Signature("question, reasoning -> answer", self.SHARED))

    def forward(self, question: str):
        reasoning = self.think(question=question).reasoning
        return self.answer(question=question, reasoning=reasoning)


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


# ---------------------------------------------------------------------------
# Instruction extraction: SBO optimizes a single shared instruction.
# ---------------------------------------------------------------------------


def test_extract_instruction_returns_single_predictor_instruction(sbo):
    program = _SingleSlotProgram(instruction="Answer the question.")
    assert sbo._extract_instruction(program) == "Answer the question."


def test_extract_instruction_accepts_unified_multi_predictor_program(sbo):
    """Multiple predictors with the SAME instruction are fine — SBO can optimize them."""
    program = _UnifiedMultiSlotProgram()
    assert sbo._extract_instruction(program) == _UnifiedMultiSlotProgram.SHARED


def test_extract_instruction_rejects_divergent_multi_predictor_program(sbo):
    """Multiple predictors with DIFFERENT instructions are out of SBO's scope."""
    program = _DivergentMultiSlotProgram()
    with pytest.raises(ValueError, match="single instruction"):
        sbo._extract_instruction(program)


def test_build_program_broadcasts_instruction_to_all_predictors(sbo):
    program = _UnifiedMultiSlotProgram()
    rebuilt = sbo._build_program(program, "New unified instruction.")

    for _, pred in rebuilt.named_predictors():
        assert pred.signature.instructions == "New unified instruction."


def test_build_program_does_not_mutate_template(sbo):
    """`_build_program` must return a fresh copy; the original template is unchanged."""
    program = _SingleSlotProgram(instruction="Original.")
    _ = sbo._build_program(program, "Rebuilt.")
    assert sbo._extract_instruction(program) == "Original."


# ---------------------------------------------------------------------------
# Judge module (dspy.Predict-backed).
# ---------------------------------------------------------------------------


CANDIDATE_INSTRUCTION = "Be concise and precise."
REFERENCE_INSTRUCTION = "Answer the question."
CRITIQUE = "Responses are too verbose."


def _judge_inputs():
    return {
        "candidate_instruction": CANDIDATE_INSTRUCTION,
        "reference_instruction": REFERENCE_INSTRUCTION,
        "critique": CRITIQUE,
    }


def test_judge_signature_fields_match_call_site():
    """The signature's input fields must match what `_judge_single` passes."""
    fields = JudgeSemanticAlignment.model_fields
    assert set(fields) == {"reference_prompt", "critique", "candidate_prompt", "score"}
    assert fields["score"].annotation is float


def test_judge_returns_parsed_score_within_range(sbo):
    """Happy path: typed float output → returned as-is."""
    lm = DummyLM([{"score": 0.5}])

    with dspy.context(lm=lm):
        score = sbo._judge_single(lm=lm, **_judge_inputs())

    assert score == pytest.approx(0.5)


def test_judge_clamps_scores_above_one(sbo):
    """A misbehaving LM that returns >1.0 must be clamped to 1.0."""
    lm = DummyLM([{"score": 5.0}])

    with dspy.context(lm=lm):
        score = sbo._judge_single(lm=lm, **_judge_inputs())

    assert score == 1.0


def test_judge_clamps_scores_below_negative_one(sbo):
    lm = DummyLM([{"score": -7.5}])

    with dspy.context(lm=lm):
        score = sbo._judge_single(lm=lm, **_judge_inputs())

    assert score == -1.0


def test_judge_returns_zero_when_module_raises(sbo, monkeypatch):
    """If the adapter exhausts retries, default to 0.0 (orthogonal) rather than crash."""

    def boom(**_kwargs):
        raise RuntimeError("adapter retries exhausted")

    monkeypatch.setattr(sbo, "_judge", boom)

    score = sbo._judge_single(lm=DummyLM([]), **_judge_inputs())

    assert score == 0.0


def test_judge_returns_zero_when_score_is_uncoercible(sbo, monkeypatch):
    """Defensive guard: if the module ever yields a non-numeric score, default to 0.0."""

    class FakeResult:
        score = "not-a-number"

    monkeypatch.setattr(sbo, "_judge", lambda **_kw: FakeResult())

    score = sbo._judge_single(lm=DummyLM([]), **_judge_inputs())

    assert score == 0.0


def test_compute_semantic_score_averages_multiple_samples(sbo, monkeypatch):
    """`_compute_semantic_score` must average `num_judge_samples` judge calls."""
    sbo.num_judge_samples = 3
    returned = iter([0.4, 0.6, 0.5])
    monkeypatch.setattr(sbo, "_judge_single", lambda *a, **kw: next(returned))

    avg = sbo._compute_semantic_score(
        CANDIDATE_INSTRUCTION, REFERENCE_INSTRUCTION, CRITIQUE, lm=DummyLM([])
    )

    assert avg == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Proposer module (dspy.Predict-backed).
# ---------------------------------------------------------------------------


def test_proposer_signature_shape():
    fields = ProposeCandidates.model_fields
    assert set(fields) == {"current_prompt", "critique", "num_candidates", "candidates"}
    assert fields["candidates"].annotation == list[str]
    assert fields["num_candidates"].annotation is int


def test_proposer_builds_one_program_per_candidate(sbo, monkeypatch):
    """Happy path: proposer returns N strings → N candidate programs are built."""
    sbo.num_candidates = 3

    fake = type(
        "FakeResult",
        (),
        {"candidates": ["Be concise.", "Be specific.", "Be friendly."]},
    )()
    monkeypatch.setattr(sbo, "_propose", lambda **_kw: fake)

    center = _SingleSlotProgram(instruction="Answer the question.")
    candidates = sbo._generate_candidates(center, critique="too vague", lm=DummyLM([]))

    assert len(candidates) == 3
    instructions = [sbo._extract_instruction(c) for c in candidates]
    assert instructions == ["Be concise.", "Be specific.", "Be friendly."]


def test_proposer_pads_with_center_when_under_quota(sbo, monkeypatch):
    """If the proposer returns fewer candidates than requested, pad with center copies."""
    sbo.num_candidates = 3

    fake = type("FakeResult", (), {"candidates": ["Only one."]})()
    monkeypatch.setattr(sbo, "_propose", lambda **_kw: fake)

    center = _SingleSlotProgram(instruction="Answer the question.")
    candidates = sbo._generate_candidates(center, critique="x", lm=DummyLM([]))

    assert len(candidates) == 3
    instructions = [sbo._extract_instruction(c) for c in candidates]
    assert instructions[0] == "Only one."
    # The remaining slots are filled with copies of the center.
    assert instructions[1] == "Answer the question."
    assert instructions[2] == "Answer the question."


def test_proposer_falls_back_to_center_on_module_exception(sbo, monkeypatch):
    """If the proposer module raises, fall back to center-program copies, don't crash."""
    sbo.num_candidates = 2

    def boom(**_kw):
        raise RuntimeError("adapter retries exhausted")

    monkeypatch.setattr(sbo, "_propose", boom)

    center = _SingleSlotProgram(instruction="Answer the question.")
    candidates = sbo._generate_candidates(center, critique="x", lm=DummyLM([]))

    assert len(candidates) == 2
    for c in candidates:
        assert sbo._extract_instruction(c) == "Answer the question."


def test_proposer_ignores_empty_and_non_string_entries(sbo, monkeypatch):
    """Whitespace-only, empty, and non-string entries are filtered before building programs."""
    sbo.num_candidates = 4

    fake = type(
        "FakeResult",
        (),
        {"candidates": ["Valid edit.", "   ", "", None, 42, "Another valid edit."]},
    )()
    monkeypatch.setattr(sbo, "_propose", lambda **_kw: fake)

    center = _SingleSlotProgram(instruction="Answer the question.")
    candidates = sbo._generate_candidates(center, critique="x", lm=DummyLM([]))

    assert len(candidates) == 4
    instructions = [sbo._extract_instruction(c) for c in candidates]
    # First two come from the proposer, the rest are padded center copies.
    assert instructions[0] == "Valid edit."
    assert instructions[1] == "Another valid edit."
    assert instructions[2] == "Answer the question."
    assert instructions[3] == "Answer the question."


def test_proposer_handles_none_candidates_field(sbo, monkeypatch):
    """If the proposer returns `candidates=None`, treat as empty and pad."""
    sbo.num_candidates = 2

    fake = type("FakeResult", (), {"candidates": None})()
    monkeypatch.setattr(sbo, "_propose", lambda **_kw: fake)

    center = _SingleSlotProgram(instruction="Answer the question.")
    candidates = sbo._generate_candidates(center, critique="x", lm=DummyLM([]))

    assert len(candidates) == 2
    for c in candidates:
        assert sbo._extract_instruction(c) == "Answer the question."


# ---------------------------------------------------------------------------
# Critique modules (DiagnoseWeakness, DiagnoseFailedCandidate).
# ---------------------------------------------------------------------------


def test_critique_signature_shapes():
    weakness_fields = DiagnoseWeakness.model_fields
    assert set(weakness_fields) == {"current_prompt", "failure_examples", "critique"}
    assert weakness_fields["critique"].annotation is str

    failed_fields = DiagnoseFailedCandidate.model_fields
    assert set(failed_fields) == {
        "candidate_prompt",
        "failure_context",
        "current_loss",
        "target_loss",
        "critique",
    }
    assert failed_fields["current_loss"].annotation is float
    assert failed_fields["target_loss"].annotation is float


def test_format_failures_renders_expected_block(sbo):
    rendered = sbo._format_failures([
        {"input": "question: q1", "expected": "answer: a", "predicted": "answer: b", "score": 0.25},
        {"input": "question: q2", "expected": "answer: c", "predicted": "answer: d", "score": 0.10},
    ])

    assert rendered == (
        "Example 1:\nInput: question: q1\nExpected: answer: a\n"
        "Predicted: answer: b\nScore: 0.25\n\n"
        "Example 2:\nInput: question: q2\nExpected: answer: c\n"
        "Predicted: answer: d\nScore: 0.10"
    )


def test_sample_failures_only_collects_below_threshold(sbo, monkeypatch):
    """Only examples scoring < threshold should be returned as failures."""
    examples = [
        Example(question="good", answer="g").with_inputs("question"),
        Example(question="bad", answer="x").with_inputs("question"),
    ]

    program = _SingleSlotProgram()
    monkeypatch.setattr(
        program,
        "forward",
        lambda question: Prediction(answer=f"pred-{question}"),
    )
    sbo.metric = lambda ex, pred, _trace: 1.0 if ex.question == "good" else 0.0

    failures = sbo._sample_failures(program, examples, sample_size=2, score_threshold=0.8)

    assert len(failures) == 1
    assert failures[0]["input"] == "question: bad"
    assert failures[0]["score"] == 0.0


def test_sample_failures_skips_examples_that_raise(sbo, monkeypatch):
    """If a program call raises, the example is dropped silently — never propagated."""
    examples = [
        Example(question="ok", answer="a").with_inputs("question"),
        Example(question="explodes", answer="b").with_inputs("question"),
    ]

    program = _SingleSlotProgram()

    def maybe_boom(question):
        if question == "explodes":
            raise RuntimeError("program crashed")
        return Prediction(answer="pred")

    monkeypatch.setattr(program, "forward", maybe_boom)
    sbo.metric = lambda *a, **kw: 0.0

    failures = sbo._sample_failures(program, examples, sample_size=2)
    inputs = sorted(f["input"] for f in failures)

    assert inputs == ["question: ok"]


def test_generate_critique_returns_no_failure_message_when_all_pass(sbo, monkeypatch):
    """If sampling finds no failures, return the canned 'performing well' message
    and skip the LM call entirely."""
    monkeypatch.setattr(sbo, "_sample_failures", lambda *a, **kw: [])

    def must_not_call(**_kw):
        raise AssertionError("LM module must not be invoked when there are no failures")

    monkeypatch.setattr(sbo, "_critique", must_not_call)

    critique = sbo._generate_critique(
        program=_SingleSlotProgram(),
        examples=[Example(question="q", answer="a").with_inputs("question")],
        instruction="Answer the question.",
        lm=DummyLM([]),
    )

    assert critique == "The prompt is performing well on the given examples."


def test_generate_critique_returns_module_output(sbo, monkeypatch):
    """Happy path: critique module is called with the formatted prompt/failures and its
    stripped output is returned."""
    captured: dict = {}

    monkeypatch.setattr(
        sbo,
        "_sample_failures",
        lambda *a, **kw: [
            {"input": "i", "expected": "e", "predicted": "p", "score": 0.1}
        ],
    )

    def fake_critique(**kwargs):
        captured.update(kwargs)
        return type("R", (), {"critique": "  Too vague about formatting.  "})()

    monkeypatch.setattr(sbo, "_critique", fake_critique)

    out = sbo._generate_critique(
        program=_SingleSlotProgram(),
        examples=[Example(question="q", answer="a").with_inputs("question")],
        instruction="Answer the question.",
        lm=DummyLM([]),
    )

    assert out == "Too vague about formatting."
    assert captured["current_prompt"] == "Answer the question."
    assert "Score: 0.10" in captured["failure_examples"]


def test_generate_critique_falls_back_when_module_raises(sbo, monkeypatch):
    monkeypatch.setattr(
        sbo,
        "_sample_failures",
        lambda *a, **kw: [
            {"input": "i", "expected": "e", "predicted": "p", "score": 0.1}
        ],
    )

    def boom(**_kw):
        raise RuntimeError("adapter retries exhausted")

    monkeypatch.setattr(sbo, "_critique", boom)

    out = sbo._generate_critique(
        program=_SingleSlotProgram(),
        examples=[Example(question="q", answer="a").with_inputs("question")],
        instruction="p",
        lm=DummyLM([]),
    )

    assert out == "The prompt failed on some examples; refine its specificity."


def test_generate_failure_critique_passes_losses_to_module(sbo, monkeypatch):
    """The failure-critique module must receive the typed loss values, not strings."""
    captured: dict = {}

    monkeypatch.setattr(sbo, "_evaluate_program", lambda *a, **kw: 0.6)
    monkeypatch.setattr(sbo, "_sample_failures", lambda *a, **kw: [])

    def fake_failure_critique(**kwargs):
        captured.update(kwargs)
        return type("R", (), {"critique": "Candidate ignored the prior critique."})()

    monkeypatch.setattr(sbo, "_failure_critique", fake_failure_critique)

    out = sbo._generate_failure_critique(
        program=_SingleSlotProgram(),
        examples=[Example(question="q", answer="a").with_inputs("question")],
        instruction="p",
        target_loss=0.4,
        critic_lm=DummyLM([]),
    )

    assert out == "Candidate ignored the prior critique."
    assert captured["current_loss"] == pytest.approx(0.6)
    assert captured["target_loss"] == pytest.approx(0.4)
    assert captured["failure_context"] == "No sampled examples available."


def test_generate_failure_critique_falls_back_when_module_raises(sbo, monkeypatch):
    monkeypatch.setattr(sbo, "_evaluate_program", lambda *a, **kw: 0.5)
    monkeypatch.setattr(sbo, "_sample_failures", lambda *a, **kw: [])

    def boom(**_kw):
        raise RuntimeError("adapter retries exhausted")

    monkeypatch.setattr(sbo, "_failure_critique", boom)

    out = sbo._generate_failure_critique(
        program=_SingleSlotProgram(),
        examples=[Example(question="q", answer="a").with_inputs("question")],
        instruction="p",
        target_loss=0.3,
        critic_lm=DummyLM([]),
    )

    assert out == (
        "The candidate did not improve loss; revisit how it addresses the prior critique."
    )


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
