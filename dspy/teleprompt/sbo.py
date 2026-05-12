"""
Semantic Bundle Optimization (SBO) for DSPy.

Based on: "Semantic Bundle Methods: Rigorous Prompt Optimization via Discrete-Continuous Relaxation"

SBO addresses limit cycles and catastrophic forgetting in greedy prompt optimization by maintaining
a "bundle" of historical critiques and using them to construct a cutting-plane model of the objective.
"""

import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

import dspy
from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


@dataclass
class BundleEntry:
    """A single entry in the optimization bundle."""
    instruction: str  # The single instruction text being optimized.
    loss: float  # F̃(p_i) - smoothed/robust loss
    critique: str  # Textual critique c_i
    iteration: int  # When this was added


@dataclass
class SBOResult:
    """Results from SBO optimization."""
    best_program: Module
    bundle: list[BundleEntry]
    best_idx: int
    val_scores: list[float]
    total_iterations: int
    num_serious_steps: int
    num_null_steps: int

#Nan: scoring only based on prompt without context maybe not be appropriate?
class JudgeSemanticAlignment(dspy.Signature):
    """Quantify how well a Candidate Prompt addresses the Critique relative to the Reference Prompt.

    Scoring rubric:
    +1.0 (Strong Descent):    Candidate completely resolves the issue.
    +0.5 (Weak Descent):      Candidate partially fixes the issue.
     0.0 (Orthogonal):        Candidate ignores the critique.
    -0.5 (Weak Regression):   Candidate slightly worsens the issue.
    -1.0 (Strong Regression): Candidate explicitly violates the critique.

    Be objective. Output a single number in [-1.0, 1.0].
    """

    reference_prompt: str = dspy.InputField(desc="The current prompt being compared against.")
    critique: str = dspy.InputField(desc="The weakness identified in the reference prompt.")
    candidate_prompt: str = dspy.InputField(desc="A proposed alternative prompt.")
    score: float = dspy.OutputField(desc="A number in [-1.0, 1.0] following the rubric.")


class ProposeCandidates(dspy.Signature):
    """Generate distinct local edits of the current prompt that address the critique.

    Each candidate must:
    1. Make local edits only — do NOT rewrite the prompt from scratch; keep structure similar.
    2. Address the critique directly.
    3. Be meaningfully different from the other candidates.
    4. Contain only the candidate instruction text — no placeholders, headers, or meta-commentary.
    """

    current_prompt: str = dspy.InputField(desc="The instruction text currently being optimized.")
    critique: str = dspy.InputField(desc="The weakness the candidates must address.")
    num_candidates: int = dspy.InputField(desc="Exact number of distinct candidate variations to return.")
    candidates: list[str] = dspy.OutputField(
        desc="Candidate instruction texts. Length should equal num_candidates."
    )


class DiagnoseWeakness(dspy.Signature):
    """You are an expert prompt engineer. Analyze the current prompt and failure
    cases to identify the SINGLE most critical weakness.

    Steps:
    1. Analyze why the prompt failed on the given examples.
    2. Formulate a specific, actionable critique (e.g., "The prompt is too vague
       about output formatting").
    3. Do NOT propose a new prompt — state the critique only.
    """

    current_prompt: str = dspy.InputField(desc="The prompt being analyzed.")
    failure_examples: str = dspy.InputField(desc="Formatted failure cases with input/expected/predicted/score.")
    critique: str = dspy.OutputField(
        desc="A specific, actionable weakness in the current prompt. Do not propose a fix."
    )


class DiagnoseFailedCandidate(dspy.Signature):
    """You are an expert prompt engineer. A candidate prompt was tested but did not
    improve performance. Explain why and provide a specific, actionable critique
    that would guide the next iteration.

    Loss interpretation: lower is better. The candidate needed to achieve
    `current_loss < target_loss` and did not. Analyze why this happened.
    """

    candidate_prompt: str = dspy.InputField(desc="The candidate prompt that failed to improve performance.")
    failure_context: str = dspy.InputField(desc="Sampled failure cases, or a note that none were available.")
    current_loss: float = dspy.InputField(desc="Loss the candidate achieved (lower is better).")
    target_loss: float = dspy.InputField(desc="Loss the candidate needed to beat (current_loss must be lower).")
    critique: str = dspy.OutputField(
        desc="Why the candidate failed and how to address it. Do not propose a fix."
    )


@experimental(version="3.1.0")
class SemanticBundleOptimization(Teleprompter):
    """
    Semantic Bundle Optimization (SBO) - A rigorous prompt optimization framework.

    Unlike greedy methods (GEPA, OPRO) that only consider the latest critique, SBO maintains
    a bundle of historical constraints to prevent limit cycles and catastrophic forgetting.

    Scope: SBO optimizes a **single instruction string** which is then broadcast to every
    predictor in the program. Programs whose predictors carry divergent task instructions
    are rejected with a ValueError — use a per-predictor optimizer (COPRO / MIPRO / GEPA)
    for that case.

    Key Components:
    - **Judge**: Semantic inner product Ŝ_J(p, p_ref, c) - scores alignment with critique
    - **Proposer**: Generates candidate variations addressing critique
    - **Verifier**: Filters candidates by cumulative violation against full bundle
    - **Serious/Null Steps**: Rigorous acceptance criterion for stability

    Args:
        metric: Evaluation function taking (example, prediction, trace) -> float or ScoreWithFeedback
        judge_lm: Language model for the semantic judge (default: main LM)
        proposer_lm: Language model for generating candidate prompts (default: main LM)
        critic_lm: Language model for generating critiques (default: main LM)
        num_candidates: Number of candidates to generate per iteration (N in paper)
        num_judge_samples: Number of judge samples for Monte Carlo averaging (J in paper)
        descent_param: Descent parameter m ∈ (0,1) for serious step test
        lambda_init: Initial sensitivity parameter λ_0
        lambda_min: Minimum allowed λ
        lambda_max: Maximum allowed λ
        lambda_gamma: EMA smoothing factor for λ updates
        tau_margin: Margin parameter for slack formulation in verifier
        max_iterations: Maximum optimization iterations
        max_null_steps: Maximum consecutive null steps before termination
        temperature: Temperature for LM calls
        track_stats: Whether to track detailed statistics
    """

    def __init__(
        self,
        metric: Callable,
        judge_lm: Optional[LM] = None,
        proposer_lm: Optional[LM] = None,
        critic_lm: Optional[LM] = None,
        num_candidates: int = 5,
        num_judge_samples: int = 3,
        descent_param: float = 0.1,
        lambda_init: float = 1.0,
        lambda_min: float = 0.1,
        lambda_max: float = 10.0,
        lambda_gamma: float = 0.3,
        tau_margin: float = 0.5,
        max_iterations: int = 50,
        max_null_steps: int = 5,
        temperature: float = 0.7,
        track_stats: bool = True,
        eval_num_threads: int = 1,
    ):
        super().__init__()
        self.metric = metric
        self.judge_lm = judge_lm
        self.proposer_lm = proposer_lm
        self.critic_lm = critic_lm
        self.num_candidates = num_candidates
        self.num_judge_samples = num_judge_samples
        self.descent_param = descent_param
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_gamma = lambda_gamma
        self.tau_margin = tau_margin
        self.max_iterations = max_iterations
        self.max_null_steps = max_null_steps
        self.temperature = temperature
        self.track_stats = track_stats
        self.eval_num_threads = max(1, eval_num_threads)

        self._judge = dspy.Predict(JudgeSemanticAlignment)
        self._propose = dspy.Predict(ProposeCandidates)
        self._critique = dspy.Predict(DiagnoseWeakness)
        self._failure_critique = dspy.Predict(DiagnoseFailedCandidate)

        self.result: Optional[SBOResult] = None

    def _format_example_fields(self, value: Example | Prediction | Any) -> str:
        """Render the fields of a DSPy Example/Prediction for a failure block.

        Trusts the DSPy data structure to expose the right fields:
        - `ex.inputs()` already isolates the program's input fields.
        - `ex.labels()` already isolates the gold-output fields.
        - `Prediction.items()` already reflects the program signature's outputs.

        Single-field dicts drop the `key:` prefix because the surrounding
        `Input:` / `Expected:` / `Predicted:` label in the failure block
        already identifies the slot — repeating `question:` / `answer:` is
        noise that distracts small critic LMs. Multi-field dicts keep the
        per-field prefixes so the structure remains parseable.
        """
        if hasattr(value, "items"):
            try:
                items = list(value.items())
            except Exception:
                return str(value)
            if len(items) == 1:
                return str(items[0][1])
            return "\n".join(f"{key}: {field}" for key, field in items)
        return str(value)

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        **kwargs
    ) -> Module:
        """
        Optimize the student program using Semantic Bundle Optimization.

        Args:
            student: Program to optimize
            trainset: Training examples (used for critique generation)
            teacher: Unused (SBO doesn't use teacher)
            valset: Validation examples for evaluating candidates

        Returns:
            Optimized program
        """
        # Setup LMs with defaults
        judge_lm = self.judge_lm or dspy.settings.lm
        proposer_lm = self.proposer_lm or dspy.settings.lm
        critic_lm = self.critic_lm or dspy.settings.lm

        if valset is None or len(valset) == 0:
            raise ValueError("SBO requires a validation set for robust loss estimation")

        logger.info(f"Starting SBO optimization with {len(trainset)} train, {len(valset)} val examples")

        # Initialize bundle with original program
        original_instruction = self._extract_instruction(student)
        logger.info(f"\n{'='*60}")
        logger.info(f"INITIAL INSTRUCTION EXTRACTED:")
        logger.info(f"  {original_instruction!r}")
        logger.info(f"{'='*60}\n")

        original_loss = self._evaluate_program(student, valset)
        logger.info(f"Original loss on valset: {original_loss:.4f}")

        initial_critique = self._generate_critique(student, trainset, original_instruction, critic_lm)
        logger.info(f"\nINITIAL CRITIQUE:")
        logger.info(f"  {initial_critique}")
        logger.info(f"")

        bundle = [BundleEntry(
            instruction=original_instruction,
            loss=original_loss,
            critique=initial_critique,
            iteration=0
        )]

        # Initialize center and sensitivity
        center_program = student.deepcopy()
        center_idx = 0
        lambda_current = self.lambda_init

        num_serious = 0
        num_null = 0
        consecutive_null = 0
        current_critique = initial_critique

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"SBO Iteration {iteration}/{self.max_iterations}")
            logger.info(f"Center loss: {bundle[center_idx].loss:.4f}, λ: {lambda_current:.3f}")
            logger.info(f"{'='*60}")

            # Stage 1: Generate candidates (Proposer)
            logger.info(f"\nCurrent critique for candidate generation:")
            logger.info(f"  {current_critique}")

            candidates = self._generate_candidates(
                center_program,
                current_critique,
                proposer_lm
            )

            logger.info(f"\nGenerated {len(candidates)} candidate programs")
            center_instruction = bundle[center_idx].instruction
            for i, candidate in enumerate(candidates):
                candidate_instruction = self._extract_instruction(candidate)
                logger.info(f"  Candidate {i+1}: {candidate_instruction!r}")
                if candidate_instruction == center_instruction:
                    logger.warning(f"    ⚠ Candidate {i+1} is IDENTICAL to center!")
                    # TODO: add a check to see if the candidate is actually different from the center. Should we drop it?

            # Stage 2: Filter candidates (Verifier)
            best_candidate, best_candidate_instruction = self._select_best_candidate(
                candidates,
                bundle,
                lambda_current,
                judge_lm
            )

            # Evaluate robust loss of selected candidate
            candidate_loss = self._evaluate_program(best_candidate, valset)

            # Compute predicted improvement
            model_value = self._compute_model_value(
                best_candidate_instruction,
                bundle,
                lambda_current,
                judge_lm
            )
            predicted_improvement = bundle[center_idx].loss - model_value
            actual_improvement = bundle[center_idx].loss - candidate_loss

            candidate_is_unchanged = best_candidate_instruction == bundle[center_idx].instruction
            status = "[UNCHANGED]" if candidate_is_unchanged else "[MODIFIED]"

            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration} RESULTS:")
            logger.info(f"  Center loss:           {bundle[center_idx].loss:.4f}")
            logger.info(f"  Candidate loss:        {candidate_loss:.4f}")
            logger.info(f"  Model value:           {model_value:.4f}")
            logger.info(f"  Predicted improvement: {predicted_improvement:.4f}")
            logger.info(f"  Actual improvement:    {actual_improvement:.4f}")
            logger.info(f"\nBest candidate selected {status}: {best_candidate_instruction!r}")
            logger.info(f"{'='*60}\n")

            if candidate_is_unchanged:
                logger.info("Selected candidate is unchanged vs center; forcing NULL STEP.")
            if predicted_improvement <= 0:
                logger.info("Predicted improvement is non-positive; forcing NULL STEP.")

            # Descent test: Serious vs Null step
            if (
                not candidate_is_unchanged
                and predicted_improvement > 0
                and actual_improvement > 0
                and actual_improvement >= self.descent_param * predicted_improvement
            ):
                # SERIOUS STEP - Accept candidate as new center
                logger.info("✓ SERIOUS STEP: Accepting candidate as new center")

                center_program = best_candidate
                center_idx = len(bundle)  # Will be the new bundle entry
                num_serious += 1
                consecutive_null = 0

                # Update sensitivity λ
                semantic_score = self._compute_semantic_score(
                    best_candidate_instruction,
                    bundle[center_idx - 1].instruction,  # Previous center
                    bundle[center_idx - 1].critique,
                    judge_lm
                )

                if abs(semantic_score) > 1e-6:  # Avoid division by zero
                    lambda_obs = actual_improvement / abs(semantic_score)
                    lambda_current = max(
                        self.lambda_min,
                        min(
                            self.lambda_max,
                            (1 - self.lambda_gamma) * lambda_current + self.lambda_gamma * lambda_obs
                        )
                    )
                    logger.info(f"Updated λ: {lambda_current:.3f} (observed: {lambda_obs:.3f})")

                # Generate critique for new center
                critique = self._generate_critique(best_candidate, trainset, best_candidate_instruction, critic_lm)

            else:
                # NULL STEP - Reject candidate, refine model
                logger.info("✗ NULL STEP: Refinement only (candidate rejected)")
                num_null += 1
                consecutive_null += 1

                # Generate critique explaining why candidate failed
                critique = self._generate_failure_critique(
                    best_candidate,
                    trainset,
                    best_candidate_instruction,
                    target_loss=bundle[center_idx].loss,
                    critic_lm=critic_lm
                )

            # Add to bundle (both serious and null steps)
            bundle.append(BundleEntry(
                instruction=best_candidate_instruction,
                loss=candidate_loss,
                critique=critique,
                iteration=iteration
            ))
            # Advance critique signal every iteration, even during null steps.
            current_critique = critique

            # Termination check
            if consecutive_null >= self.max_null_steps:
                logger.info(f"Terminating: {self.max_null_steps} consecutive null steps")
                break

            # Only terminate on non-positive predicted improvement after a few iterations
            # This gives the optimizer time to build a useful bundle
            if iteration >= 3 and predicted_improvement <= 0:
                logger.info("Terminating: Non-positive predicted improvement (after iteration 3+)")
                break

        # Return best program from bundle
        best_idx = min(range(len(bundle)), key=lambda i: bundle[i].loss)
        best_program = self._build_program(student, bundle[best_idx].instruction)

        val_scores = [b.loss for b in bundle]

        self.result = SBOResult(
            best_program=best_program,
            bundle=bundle,
            best_idx=best_idx,
            val_scores=val_scores,
            total_iterations=iteration,
            num_serious_steps=num_serious,
            num_null_steps=num_null
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"SBO Optimization Complete")
        logger.info(f"Best loss: {bundle[best_idx].loss:.4f} (iteration {bundle[best_idx].iteration})")
        logger.info(f"Serious steps: {num_serious}, Null steps: {num_null}")
        logger.info(f"{'='*60}\n")

        return best_program

    def _extract_instruction(self, program: Module) -> str:
        """Return the single instruction SBO will optimize for this program.

        SBO optimizes one instruction string and broadcasts it to every predictor.
        Returns the *normalized* semantic instruction: DSPy auto-synthesized
        scaffolds (e.g. "Given the fields `x`, produce the fields `y`.") are
        collapsed to "" so downstream consumers (proposer, critic, judge) all
        agree on what "no task instruction set" means.

        If the program has multiple predictors with divergent task instructions,
        raise — there is no single instruction to optimize. Use a per-predictor
        optimizer (COPRO / MIPRO / GEPA) for that case.
        """
        instructions: list[str] = []
        for _, pred in program.named_predictors():
            if hasattr(pred, "signature") and hasattr(pred.signature, "instructions"):
                instructions.append(
                    self._normalize_instruction_text(pred.signature.instructions or "")
                )

        if not instructions:
            return ""

        if len(set(instructions)) > 1:
            raise ValueError(
                "SBO optimizes a single instruction shared across all predictors, "
                "but this program has predictors with divergent task instructions. "
                "Unify them, or use a multi-predictor optimizer (COPRO / MIPRO / GEPA)."
            )
        return instructions[0]

    def _normalize_instruction_text(self, instruction: str) -> str:
        """Keep only semantic instruction text; drop DSPy auto-scaffold defaults."""
        text = (instruction or "").strip()
        if not text:
            return ""

        # DSPy may synthesize structural default instructions such as:
        # "Given the fields `question`, produce the fields `answer`."
        # Treat those as empty so SBO optimizes task instructions only.
        scaffold_pattern = re.compile(
            r"^given the fields\s+`?.+?`?,\s*produce the fields\s+`?.+?`?\.?$",
            re.IGNORECASE | re.DOTALL,
        )
        if scaffold_pattern.match(text):
            return ""

        return text

    def _build_program(self, template: Module, instruction: str) -> Module:
        """Return a deepcopy of `template` with `instruction` applied to every predictor."""
        program = template.deepcopy()
        for _, pred in program.named_predictors():
            if hasattr(pred, "signature"):
                # `with_instructions()` returns a NEW signature, must reassign.
                pred.signature = pred.signature.with_instructions(instruction)
        return program

    def _evaluate_program(self, program: Module, examples: list[Example]) -> float:
        """Evaluate program on examples using the metric (robust loss estimation)."""
        total_loss = 0.0
        num_threads = max(1, self.eval_num_threads)
        logger.info(f"Evaluating program on {len(examples)} examples...")
        logger.info(f"Using {'parallel' if num_threads > 1 else 'sequential'} SBO eval (num_threads={num_threads})")

        if num_threads == 1:
            for idx, ex in enumerate(examples):
                loss, error = self._evaluate_single_example(program, ex)
                if error is not None:
                    logger.warning(f"Evaluation error on example {idx+1}: {error}")
                    logger.debug(f"  Example inputs: {ex.inputs()}")
                total_loss += loss
        else:
            indexed_losses: dict[int, tuple[float, str | None, Example]] = {}
            executor = ThreadPoolExecutor(max_workers=num_threads)
            try:
                futures = {
                    executor.submit(self._evaluate_single_example, program, ex): idx
                    for idx, ex in enumerate(examples)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    ex = examples[idx]
                    try:
                        loss, error = future.result()
                    except Exception as e:  # Defensive fallback
                        loss, error = 1.0, str(e)
                    indexed_losses[idx] = (loss, error, ex)
            except KeyboardInterrupt:
                logger.warning("SBO parallel evaluation interrupted; cancelling pending tasks...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            for idx in range(len(examples)):
                loss, error, ex = indexed_losses[idx]
                if error is not None:
                    logger.warning(f"Evaluation error on example {idx+1}: {error}")
                    logger.debug(f"  Example inputs: {ex.inputs()}")
                total_loss += loss

        avg_loss = total_loss / len(examples)
        logger.info(f"Program evaluation complete: avg_loss={avg_loss:.4f}")
        return avg_loss

    def _evaluate_single_example(self, program: Module, ex: Example) -> tuple[float, str | None]:
        """Evaluate one example and return (loss, optional error string)."""
        try:
            pred = program(**ex.inputs())
            score = self.metric(ex, pred, None)
            loss = 1.0 - float(score)
            return loss, None
        except Exception as e:
            return 1.0, str(e)

    def _sample_failures(
        self,
        program: Module,
        examples: list[Example],
        sample_size: int = 3,
        score_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Run `program` on a small sample and collect cases that score below threshold.

        `Example.labels()` returns every non-input field on the example — which
        for datasets like HotPotQA includes large metadata fields (`context`,
        `gold_titles`, `id`, `type`) the program never sees or emits. Feeding
        those into the critic balloons the prompt and distracts the LM with
        irrelevant content. Scope `expected` strictly to the program signature's
        output fields so the critic only reasons about what the program is
        actually being asked to produce.
        """
        output_field_names = self._collect_output_field_names(program)
        failures: list[dict[str, Any]] = []
        for ex in random.sample(examples, min(sample_size, len(examples))):
            try:
                pred = program(**ex.inputs())
                score = self.metric(ex, pred, None)
            except Exception:
                continue
            if score < score_threshold:
                expected_fields = {
                    key: value
                    for key, value in ex.labels().items()
                    if not output_field_names or key in output_field_names
                }
                failures.append({
                    "input": self._format_example_fields(ex.inputs()),
                    "expected": self._format_example_fields(expected_fields),
                    "predicted": self._format_example_fields(pred),
                    "score": float(score),
                })
        return failures

    def _collect_output_field_names(self, program: Module) -> set[str]:
        """Union of output-field names declared by every predictor in `program`.

        Used to filter dataset Examples down to fields the program signature
        actually emits, so failure blocks don't include dataset-only metadata.
        """
        names: set[str] = set()
        for _, pred in program.named_predictors():
            if hasattr(pred, "signature") and hasattr(pred.signature, "output_fields"):
                names.update(pred.signature.output_fields.keys())
        return names

    def _format_failures(self, failures: list[dict[str, Any]]) -> str:
        """Render failure dicts as the block consumed by critique modules."""
        return "\n\n".join(
            f"Example {i+1}:\nInput: {f['input']}\nExpected: {f['expected']}\n"
            f"Predicted: {f['predicted']}\nScore: {f['score']:.2f}"
            for i, f in enumerate(failures)
        )

    def _generate_critique(
        self,
        program: Module,
        examples: list[Example],
        instruction: str,
        lm: LM
    ) -> str:
        """Generate critique identifying weaknesses in the program's instruction."""
        failures = self._sample_failures(program, examples)

        if not failures:
            return "The prompt is performing well on the given examples."

        failures_text = self._format_failures(failures)

        with dspy.context(lm=lm, temperature=self.temperature):
            logger.info(f"Critique generation - current instruction: {instruction}")
            logger.info(f"Critique generation - sampled failures: {failures_text}")
            try:
                result = self._critique(
                    current_prompt=instruction,
                    failure_examples=failures_text,
                )
            except Exception as e:
                logger.warning("Critique module failed (%s); returning generic critique.", e)
                return "The prompt failed on some examples; refine its specificity."

        logger.info(f"Critique result: {result.critique}")
        return result.critique.strip()

    def _generate_failure_critique(
        self,
        program: Module,
        examples: list[Example],
        instruction: str,
        target_loss: float,
        critic_lm: LM
    ) -> str:
        """Generate critique explaining why a candidate's instruction failed to improve."""
        # Use a bounded sample for failure critique to reduce null-step overhead.
        eval_examples = random.sample(examples, min(3, len(examples)))
        current_loss = self._evaluate_program(program, eval_examples)

        failures = self._sample_failures(program, eval_examples)
        failures_text = self._format_failures(failures) if failures else "No sampled examples available."

        with dspy.context(lm=critic_lm, temperature=self.temperature):
            try:
                result = self._failure_critique(
                    candidate_prompt=instruction,
                    failure_context=failures_text,
                    current_loss=current_loss,
                    target_loss=target_loss,
                )
            except Exception as e:
                logger.warning("Failure-critique module failed (%s); returning generic critique.", e)
                return "The candidate did not improve loss; revisit how it addresses the prior critique."

        logger.info(f"Failure-critique result: {result.critique}")
        return result.critique.strip()

    def _generate_candidates(
        self,
        center_program: Module,
        critique: str,
        lm: LM
    ) -> list[Module]:
        """Generate N candidate variations addressing the critique (Proposer)."""
        center_instruction = self._extract_instruction(center_program)
        logger.info(f"Center instruction: {center_instruction!r}")

        prompt_text = center_instruction or "(no task-specific instruction currently set)"

        try:
            with dspy.context(lm=lm, temperature=self.temperature):
                result = self._propose(
                    current_prompt=prompt_text,
                    critique=critique,
                    num_candidates=self.num_candidates,
                )
            raw_candidates = list(result.candidates or [])
        except Exception as e:
            logger.warning("Proposer module failed (%s); will fall back to center program.", e)
            raw_candidates = []

        cleaned_candidates = [
            text.strip()
            for text in raw_candidates
            if isinstance(text, str) and text.strip()
        ]
        for idx, candidate_text in enumerate(cleaned_candidates, 1):
            logger.info(f"Parsed candidate {idx}:\n{candidate_text}")

        candidates: list[Module] = [
            self._build_program(center_program, candidate_text)
            for candidate_text in cleaned_candidates[:self.num_candidates]
        ]

        if not candidates:
            logger.warning("Proposer returned no usable candidates; using center program.")
            candidates.append(center_program.deepcopy())

        while len(candidates) < self.num_candidates:
            logger.warning(
                "Only %d candidates available, padding with center program copy.",
                len(candidates),
            )
            candidates.append(center_program.deepcopy())

        num_from_proposer = min(len(cleaned_candidates), self.num_candidates)
        logger.info(
            "Candidate generation summary: returning %d candidates (%d from proposer, %d padded).",
            self.num_candidates,
            num_from_proposer,
            self.num_candidates - num_from_proposer,
        )
        return candidates[:self.num_candidates]

    def _compute_semantic_score(
        self,
        candidate_instruction: str,
        reference_instruction: str,
        critique: str,
        lm: LM
    ) -> float:
        """
        Compute smoothed semantic score Ŝ_J(p, p_ref, c).

        Returns value in [-1, 1] indicating alignment with critique.
        """
        scores = []

        for _ in range(self.num_judge_samples):
            score = self._judge_single(candidate_instruction, reference_instruction, critique, lm)
            scores.append(score)

        return sum(scores) / len(scores)

    def _judge_single(
        self,
        candidate_instruction: str,
        reference_instruction: str,
        critique: str,
        lm: LM
    ) -> float:
        """Single judge evaluation (returns clamped to [-1, 1])."""
        with dspy.context(lm=lm, temperature=0.0):  # Deterministic
            try:
                result = self._judge(
                    reference_prompt=reference_instruction,
                    critique=critique,
                    candidate_prompt=candidate_instruction,
                )
            except Exception as e:
                # Adapter exhausted retries on a malformed response. Treat as orthogonal.
                logger.warning("Judge module failed; defaulting score to 0.0: %s", e)
                return 0.0

        try:
            score = float(result.score)
        except (TypeError, ValueError):
            logger.warning("Failed to coerce judge score %r; defaulting to 0.0.", result.score)
            return 0.0

        return max(-1.0, min(1.0, score))

    def _select_best_candidate(
        self,
        candidates: list[Module],
        bundle: list[BundleEntry],
        lambda_current: float,
        lm: LM
    ) -> tuple[Module, str]:
        """
        Select best candidate by minimizing cumulative semantic violation (Verifier).

        Returns: (best_program, best_instruction)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"VERIFIER: Evaluating {len(candidates)} candidates against bundle of {len(bundle)}")
        logger.info(f"  (This requires {len(candidates) * len(bundle) * self.num_judge_samples} judge LM calls)")
        logger.info(f"{'='*60}")

        best_violation = float('inf')
        best_candidate = candidates[0]
        best_instruction = self._extract_instruction(candidates[0])

        for idx, candidate in enumerate(candidates):
            candidate_instruction = self._extract_instruction(candidate)

            # Compute cumulative violation against all bundle entries
            total_violation = 0.0
            scores_detail = []
            for entry in bundle:
                score = self._compute_semantic_score(
                    candidate_instruction,
                    entry.instruction,
                    entry.critique,
                    lm
                )
                # Hinge loss: max(0, τ - score)
                violation = max(0, self.tau_margin - score)
                total_violation += violation
                scores_detail.append((score, violation))

            logger.info(f"  Candidate {idx+1}: total_violation={total_violation:.4f}")
            for i, (score, viol) in enumerate(scores_detail):
                logger.info(f"    vs bundle[{i}]: score={score:.3f}, violation={viol:.3f}")

            if total_violation < best_violation:
                best_violation = total_violation
                best_candidate = candidate
                best_instruction = candidate_instruction

        logger.info(f"\n✓ Selected candidate with violation: {best_violation:.4f}")
        logger.info(f"{'='*60}\n")
        return best_candidate, best_instruction

    def _compute_model_value(
        self,
        instruction: str,
        bundle: list[BundleEntry],
        lambda_current: float,
        lm: LM
    ) -> float:
        """
        Compute cutting-plane model M_k(p) = max_i {F̃_i - λ·Ŝ_J(p, p_i, c_i)}.
        """
        model_value = float('-inf')

        for entry in bundle:
            score = self._compute_semantic_score(instruction, entry.instruction, entry.critique, lm)
            value = entry.loss - lambda_current * score
            model_value = max(model_value, value)

        return model_value
