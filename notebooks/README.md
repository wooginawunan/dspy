# HotPotQA GEPA Benchmark

This directory contains scripts for benchmarking DSPy optimizers on the HotPotQA dataset.

## Quick Start

```bash
# Baseline evaluation (no optimization)
python gepa_hotpotqa.py --optimizer baseline --program context --dev_size 100

# GEPA optimization
python gepa_hotpotqa.py --optimizer gepa --program mlflow_base_prompt --train_size 100 --dev_size 50

# MIPROv2 optimization
python gepa_hotpotqa.py --optimizer mipro --program context --auto light
```

## Parameter Selection Guide

### Key Parameters for GEPA

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--reflection_minibatch_size` | 3 | **1-2** for small models | Number of examples in each reflection prompt. Larger = longer prompts. |
| `--train_size` | 50 | 100-300 | Training examples for optimization. More = better coverage but slower. |
| `--dev_size` | 100 | 10-50 | Validation examples for tracking scores. Smaller = faster iterations. |
| `--no_cache` | False | Use when debugging | Disable LM response caching for fresh runs. |

### Context Window Considerations

⚠️ **Important**: HotPotQA examples contain long context passages. The reflection prompt sent to the reflection LM includes:
- Current instruction
- Multiple examples with full context, question, answer, and feedback

**Prompt size estimation**:
- Each HotPotQA example in the reflection prompt: ~5,000-15,000 characters
- With `reflection_minibatch_size=3`: prompts can exceed 20,000+ characters (~5,000-7,000 tokens)

**Recommendations by model size**:

| Model Size | `reflection_minibatch_size` | Notes |
|------------|----------------------------|-------|
| 4B-7B | 1 | Keep prompts short; may still truncate on long examples |
| 13B-30B | 1-2 | Better handling of longer contexts |
| 70B+ / API | 2-3 | Can handle full reflection prompts |

### Program Types

| Program | Uses Context | Description |
|---------|--------------|-------------|
| `naive` | No | Direct question → answer |
| `reasoning` | No | Two-stage: think → answer |
| `context` | Yes | Uses gold passages from HotPotQA |
| `reasoning_context` | Yes | Two-stage with context |
| `mlflow_base_prompt` | Yes | Pre-defined instruction template |
| `mlflow_v2` | Yes | Uses `.with_instructions()` method |
| `simple_context` | Yes | Minimal signature, lets GEPA propose from scratch |

### Troubleshooting

#### Empty or truncated instruction proposals

**Symptom**: GEPA logs show `Proposed new text for answer:` with empty or very short text.

**Causes**:
1. Reflection prompt too long for model's effective context
2. Model returns only code fences (````) without content

**Solutions**:
- Reduce `--reflection_minibatch_size` to 1
- Use a larger model for reflection (`--reflection_lm`)
- Add `max_tokens=4096` to the reflection LM (already configured)

#### "All subsample scores perfect. Skipping."

**Symptom**: GEPA skips reflection because the minibatch scored perfectly.

**Cause**: With small `reflection_minibatch_size`, you may randomly sample only "easy" examples.

**Solutions**:
- Increase `--train_size` for more diverse sampling
- Set `skip_perfect_score=False` in GEPA config (requires code change)

#### Proposals don't improve scores

**Symptom**: New instructions score worse than baseline on validation set.

**Cause**: The reflection LM may propose overly specific or incorrect heuristics based on the small minibatch.

**Solutions**:
- Increase `--train_size` for better coverage
- Use a stronger reflection model
- Run longer with more `max_metric_calls`

## Example Commands

### Small local model (qwen3:4b)

```bash
# Minimal settings for 4B model
python gepa_hotpotqa.py \
    --optimizer gepa \
    --program simple_context \
    --model ollama_chat/qwen3:4b-instruct \
    --reflection_lm ollama_chat/qwen3:4b-instruct \
    --reflection_minibatch_size 1 \
    --train_size 50 \
    --dev_size 10 \
    --no_cache
```

### Larger model or API

```bash
# Full settings for larger models
python gepa_hotpotqa.py \
    --optimizer gepa \
    --program mlflow_base_prompt \
    --model gpt-4o-mini \
    --reflection_lm gpt-4o \
    --reflection_minibatch_size 3 \
    --train_size 200 \
    --dev_size 50
```

## Output Interpretation

### GEPA Candidate Inspection

After optimization, the script prints all proposed candidates:

```
============================================================
GEPA CANDIDATE INSPECTION
============================================================
Total candidates explored: 5
Best candidate index: 0
Best validation score: 0.7263

[Rank 1] Candidate 0 — Score: 0.7263 ★ BEST
  [answer]:
    You are a question answering assistant...

[Rank 2] Candidate 1 — Score: 0.4164
  [answer]:
    Given the fields `context`, `question`, produce the fields `answer`.
```

- **Candidate 0** is always the original/baseline
- If Candidate 0 remains best, optimization found no improvements
- Candidates with default DSPy signature (`Given the fields...`) indicate empty proposals

### Score Distribution

```
SCORE DISTRIBUTION
  Min:    0.3520
  Max:    0.5411
  Mean:   0.3917
```

- **High variance** (large Max-Min gap): proposals are exploring diverse strategies
- **Low mean with high max**: some proposals work well, others don't
- **All similar scores**: optimization is stuck or baseline is already optimal

## Files

- `gepa_hotpotqa.py` - Main benchmark script
- `programs.py` - DSPy program/module definitions
- `hotpotqa_review.py` - Dataset inspection utilities

