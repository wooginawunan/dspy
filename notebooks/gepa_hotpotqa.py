"""
HotPotQA Benchmark Script for DSPy Optimizers

This script provides a modular framework for benchmarking different DSPy optimizers
on the HotPotQA dataset. Supports GEPA, MIPROv2, BootstrapFewShot, COPRO, and baseline comparison.

Usage:
    python gepa_hotpotqa.py --optimizer gepa --train_size 50 --dev_size 100
    python gepa_hotpotqa.py --optimizer mipro --auto light
    python gepa_hotpotqa.py --optimizer bootstrap --max_rounds 3
    python gepa_hotpotqa.py --optimizer baseline  # No optimization, just evaluate
"""

from __future__ import annotations

import argparse
import re
import string
import unicodedata
from typing import TYPE_CHECKING, Any, Callable

import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate.metrics import hotpot_f1_score

if TYPE_CHECKING:
    from dspy import Example, Module, Prediction

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL = "ollama_chat/qwen3:4b-instruct"
DEFAULT_API_BASE = "http://localhost:11434"
AUTO_BUDGET_CHOICES = ("light", "medium", "heavy")

# ============================================================================
# METRICS
# ============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and removing articles/punctuation."""
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_yesno(text: str) -> str | None:
    """Extract yes/no from a verbose answer if it starts with yes/no."""
    normalized = normalize_text(text)
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return None


def smart_hotpot_f1_score(
    prediction: str,
    ground_truth: str,
    should_extract_yesno: bool = False,
) -> float:
    """Smart HotPotQA F1 score that handles verbose yes/no answers.

    For yes/no questions, extracts the yes/no prefix from verbose predictions.
    For other questions, uses standard token-level F1 scoring.

    Args:
        prediction: Predicted answer.
        ground_truth: Reference answer.
        should_extract_yesno: Whether to extract yes/no from verbose predictions.

    Returns:
        F1 score in [0.0, 1.0].

    Example:
        >>> smart_hotpot_f1_score("No, only Cangzhou is in Hebei Province", "no")
        1.0
    """
    norm_pred = normalize_text(prediction)
    norm_gold = normalize_text(ground_truth)

    # Handle yes/no/noanswer special cases
    yesno_answers = {"yes", "no", "noanswer"}
    if norm_gold in yesno_answers and should_extract_yesno:
        extracted = extract_yesno(norm_pred)
        if extracted is not None:
            return 1.0 if extracted == norm_gold else 0.0

    return hotpot_f1_score(norm_pred, norm_gold)


def _get_answer(obj: Any, attr: str = "answer") -> str:
    """Safely extract answer attribute from an object."""
    return getattr(obj, attr, "") or ""


def _compute_f1_score(pred_answer: str, gold_answer: str | list[str]) -> float:
    """Compute F1 score, handling both single and multi-reference answers."""
    if isinstance(gold_answer, list):
        return max(smart_hotpot_f1_score(pred_answer, ans) for ans in gold_answer)
    return smart_hotpot_f1_score(pred_answer, gold_answer)


def hotpotqa_metric(example: Example, pred: Prediction, trace: Any = None) -> float:
    """Standard HotPotQA metric for evaluation.

    Uses smart F1 scoring that handles verbose yes/no answers.
    Compatible with dspy.Evaluate and most optimizers.
    """
    pred_answer = _get_answer(pred)
    gold_answer = _get_answer(example)
    return _compute_f1_score(pred_answer, gold_answer)


def hotpotqa_metric_gepa(
    gold: Example,
    pred: Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Prediction:
    """GEPA-compatible metric with feedback for reflective optimization.

    GEPA requires (gold, pred, trace, pred_name, pred_trace) signature and
    returns a Prediction with score and feedback fields.
    """
    pred_answer = _get_answer(pred)
    gold_answer = _get_answer(gold)
    score = _compute_f1_score(pred_answer, gold_answer)

    # Provide feedback for GEPA's reflection
    if score < 1.0:
        feedback = f"Expected: '{gold_answer}', Got: '{pred_answer}'. Score: {score:.2f}"
    else:
        feedback = "Perfect match!"

    return dspy.Prediction(score=score, feedback=feedback)


# ============================================================================
# DATASET
# ============================================================================


def load_dataset(
    train_seed: int = 1,
    train_size: int = 50,
    eval_seed: int = 2023,
    dev_size: int = 100,
    test_size: int = 0,
    keep_details: bool = True,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Load and prepare the HotPotQA dataset.

    Args:
        train_seed: Random seed for training split.
        train_size: Number of training examples.
        eval_seed: Random seed for evaluation split.
        dev_size: Number of validation examples.
        test_size: Number of test examples.
        keep_details: Whether to keep detailed metadata.

    Returns:
        Tuple of (train_set, val_set, test_set) with question as input.

    Note:
        Always uses hard examples (only_hard_examples=True) for proper benchmarking.
    """
    dataset = HotPotQA(
        train_seed=train_seed,
        train_size=train_size,
        eval_seed=eval_seed,
        dev_size=dev_size,
        test_size=test_size,
        only_hard_examples=True,
        keep_details=keep_details,
    )

    train_set = [ex.with_inputs("question") for ex in dataset.train]
    val_set = [ex.with_inputs("question") for ex in dataset.dev]
    test_set = [ex.with_inputs("question") for ex in dataset.test]

    print(f"Dataset loaded: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set


# ============================================================================
# MODEL SETUP
# ============================================================================


def setup_model(model_name: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE) -> dspy.LM:
    """Configure and return the language model."""
    lm = dspy.LM(model=model_name, api_base=api_base)
    dspy.configure(lm=lm)
    print(f"Model configured: {model_name}")
    return lm


# ============================================================================
# PROGRAM DEFINITION
# ============================================================================

from programs import PROGRAMS, create_program

# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_program(
    program: Module,
    dataset: list[Example],
    name: str = "",
    num_threads: int = 1,
    verbose: bool = True,
) -> Any:
    """Evaluate a program on a dataset and optionally print results.

    Args:
        program: The DSPy program to evaluate.
        dataset: List of examples to evaluate on.
        name: Name for display in output.
        num_threads: Number of parallel threads.
        verbose: Whether to print detailed results.

    Returns:
        Evaluation results object with .score and .results attributes.
    """
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=hotpotqa_metric,
        num_threads=num_threads,
        display_table=False,
        display_progress=True,
    )
    results = evaluator(program)

    if verbose:
        _print_evaluation_results(results, dataset, name)

    return results


def _print_evaluation_results(results: Any, dataset: list[Example], name: str) -> None:
    """Print detailed evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"{name} RESULTS")
    print(f"{'=' * 50}")

    scores = [score for _, _, score in results.results]
    total_score = sum(scores)
    avg_score = total_score / len(dataset) if dataset else 0
    print(f"Total Score: {total_score:.2f} / {len(dataset)} ({100 * avg_score:.1f}%)")

    for i, (example, prediction, score) in enumerate(results.results):
        question_preview = example.question[:80]
        pred_answer = getattr(prediction, "answer", str(prediction))
        print(f"\n[{i + 1}] Q: {question_preview}...")
        print(f"    Gold: {example.answer}")
        print(f"    Pred: {pred_answer}")
        print(f"    Score: {score:.3f}")

# ============================================================================
# OPTIMIZERS
# ============================================================================


def optimize_with_gepa(
    program: Module,
    train_set: list[Example],
    val_set: list[Example],
    args: argparse.Namespace,
) -> Module:
    """Optimize using GEPA optimizer with reflective feedback."""
    from dspy import GEPA

    reflection_model = args.reflection_lm or args.model
    reflection_lm = dspy.LM(
        model=reflection_model,
        api_base=args.api_base,
        temperature=1.0,
    )

    optimizer = GEPA(
        metric=hotpotqa_metric_gepa,
        reflection_lm=reflection_lm,
        max_metric_calls=100,
        num_threads=args.num_threads,
        track_stats=True,
        reflection_minibatch_size=args.reflection_minibatch_size,
    )

    print(f"\nOptimizing with GEPA (auto={args.auto})...")
    return optimizer.compile(program, trainset=train_set, valset=val_set)


def optimize_with_mipro(
    program: Module,
    train_set: list[Example],
    val_set: list[Example],
    args: argparse.Namespace,
) -> Module:
    """Optimize using MIPROv2 optimizer."""
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=hotpotqa_metric,
        auto=args.auto,
        num_threads=args.num_threads,
    )

    print(f"\nOptimizing with MIPROv2 (auto={args.auto})...")
    return optimizer.compile(program, trainset=train_set, valset=val_set)


def optimize_with_bootstrap(
    program: Module,
    train_set: list[Example],
    val_set: list[Example],
    args: argparse.Namespace,
) -> Module:
    """Optimize using BootstrapFewShot optimizer."""
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(
        metric=hotpotqa_metric,
        max_rounds=args.max_rounds,
        max_bootstrapped_demos=args.max_demos,
    )

    print(f"\nOptimizing with BootstrapFewShot (max_rounds={args.max_rounds})...")
    return optimizer.compile(program, trainset=train_set)


def optimize_with_copro(
    program: Module,
    train_set: list[Example],
    val_set: list[Example],
    args: argparse.Namespace,
) -> Module:
    """Optimize using COPRO optimizer."""
    from dspy.teleprompt import COPRO

    optimizer = COPRO(
        metric=hotpotqa_metric,
        breadth=args.breadth,
        depth=args.depth,
    )

    print(f"\nOptimizing with COPRO (breadth={args.breadth}, depth={args.depth})...")
    return optimizer.compile(program, trainset=train_set, eval_kwargs={"devset": val_set})


# Registry of available optimizers
OPTIMIZERS: dict[str, Callable | None] = {
    "gepa": optimize_with_gepa,
    "mipro": optimize_with_mipro,
    "bootstrap": optimize_with_bootstrap,
    "copro": optimize_with_copro,
    "baseline": None,
}


# ============================================================================
# MAIN BENCHMARK LOGIC
# ============================================================================


def _print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a section header."""
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def _print_prompt_comparison(baseline_program: Module, optimized_program: Module) -> None:
    """Print a comparison of baseline vs optimized prompts (full text)."""
    print("\n" + "-" * 60)
    print("PROMPT COMPARISON")
    print("-" * 60)

    baseline_predictors = dict(baseline_program.named_predictors())
    optimized_predictors = dict(optimized_program.named_predictors())

    for pred_name, baseline_pred in baseline_predictors.items():
        optimized_pred = optimized_predictors[pred_name]
        baseline_instr = baseline_pred.signature.instructions
        optimized_instr = optimized_pred.signature.instructions

        print(f"\n[PREDICTOR: {pred_name}]")
        print(f"  Baseline:")
        print(f"    {baseline_instr}")
        print(f"  Optimized:")
        print(f"    {optimized_instr}")

        status = "[CHANGED]" if baseline_instr != optimized_instr else "[UNCHANGED]"
        print(f"  {status}")

    print("-" * 60)


def _print_comparison_summary(
    baseline_results: Any,
    optimized_results: Any,
    dataset_size: int,
) -> None:
    """Print summary comparing baseline and optimized results."""
    # Compute actual scores from results to avoid scaling issues
    baseline_scores = [score for _, _, score in baseline_results.results]
    optimized_scores = [score for _, _, score in optimized_results.results]

    baseline_total = sum(baseline_scores)
    optimized_total = sum(optimized_scores)

    baseline_pct = (baseline_total / dataset_size * 100) if dataset_size > 0 else 0
    optimized_pct = (optimized_total / dataset_size * 100) if dataset_size > 0 else 0

    improvement = optimized_total - baseline_total
    pct_improvement = (improvement / baseline_total * 100) if baseline_total > 0 else 0

    _print_section("COMPARISON SUMMARY")
    print(f"Baseline:    {baseline_total:.2f} / {dataset_size} ({baseline_pct:.1f}%)")
    print(f"Optimized:   {optimized_total:.2f} / {dataset_size} ({optimized_pct:.1f}%)")
    print(f"Improvement: {improvement:+.2f} ({pct_improvement:+.1f}% relative)")


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[Any, Any | None, Module | None]:
    """Run the benchmark with specified optimizer.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (baseline_results, optimized_results, optimized_program).
        optimized_results and optimized_program are None for baseline runs.
    """
    # Load dataset
    train_set, val_set, _ = load_dataset(
        train_seed=args.train_seed,
        train_size=args.train_size,
        eval_seed=args.eval_seed,
        dev_size=args.dev_size,
        test_size=args.test_size,
    )

    # Setup model and program
    setup_model(args.model, args.api_base)
    program = create_program(args.program)

    # Evaluate baseline
    _print_section("BASELINE EVALUATION")
    baseline_results = evaluate_program(
        program, val_set, name="Baseline", num_threads=args.num_threads
    )

    # Return early if no optimization requested
    if args.optimizer == "baseline":
        return baseline_results, None, None

    # Get optimizer function
    optimizer_fn = OPTIMIZERS.get(args.optimizer)
    if optimizer_fn is None:
        print(f"Unknown optimizer: {args.optimizer}")
        return baseline_results, None, None

    # Run optimization
    _print_section(f"OPTIMIZATION: {args.optimizer.upper()}")
    optimized_program = optimizer_fn(program, train_set, val_set, args)

    # Show prompt changes
    _print_prompt_comparison(program, optimized_program)

    # Evaluate optimized program
    _print_section("OPTIMIZED EVALUATION")
    optimized_results = evaluate_program(
        optimized_program, val_set, name="Optimized", num_threads=args.num_threads
    )

    # Print comparison
    _print_comparison_summary(baseline_results, optimized_results, len(val_set))

    return baseline_results, optimized_results, optimized_program


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HotPotQA Optimizer Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--train_seed", type=int, default=1, help="Random seed for train split")
    dataset_group.add_argument("--train_size", type=int, default=50, help="Number of training examples")
    dataset_group.add_argument("--eval_seed", type=int, default=2023, help="Random seed for eval split")
    dataset_group.add_argument("--dev_size", type=int, default=100, help="Number of validation examples")
    dataset_group.add_argument("--test_size", type=int, default=0, help="Number of test examples")

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    model_group.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help="API base URL")

    # Program arguments
    program_group = parser.add_argument_group("Program")
    program_group.add_argument(
        "--program",
        type=str,
        default="naive",
        choices=list(PROGRAMS.keys()),
        help="Program type: 'naive' (direct QA) or 'reasoning' (two-stage reasoning-first)",
    )

    # General optimizer arguments
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument(
        "--optimizer",
        type=str,
        default="baseline",
        choices=list(OPTIMIZERS.keys()),
        help="Optimizer to use",
    )
    opt_group.add_argument(
        "--auto",
        type=str,
        default="light",
        choices=list(AUTO_BUDGET_CHOICES),
        help="Auto budget for GEPA/MIPROv2",
    )
    opt_group.add_argument("--num_threads", type=int, default=1, help="Number of threads")

    # GEPA-specific arguments
    gepa_group = parser.add_argument_group("GEPA")
    gepa_group.add_argument(
        "--reflection_lm",
        type=str,
        default=None,
        help="Reflection LM for GEPA (defaults to --model if not set)",
    )
    gepa_group.add_argument(
        "--reflection_minibatch_size",
        type=int,
        default=3,
        help="GEPA reflection minibatch size",
    )

    # BootstrapFewShot-specific arguments
    bootstrap_group = parser.add_argument_group("BootstrapFewShot")
    bootstrap_group.add_argument("--max_rounds", type=int, default=3, help="Max bootstrap rounds")
    bootstrap_group.add_argument("--max_demos", type=int, default=4, help="Max bootstrapped demos")

    # COPRO-specific arguments
    copro_group = parser.add_argument_group("COPRO")
    copro_group.add_argument("--breadth", type=int, default=10, help="Search breadth")
    copro_group.add_argument("--depth", type=int, default=3, help="Search depth")

    return parser.parse_args()


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    """Main entry point for the benchmark script."""
    args = parse_args()

    _print_section(f"HotPotQA Benchmark - Optimizer: {args.optimizer.upper()}")
    print(f"Config: train={args.train_size}, val={args.dev_size}, model={args.model}")

    run_benchmark(args)


if __name__ == "__main__":
    main()
