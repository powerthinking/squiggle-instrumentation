# squiggle-instrumentation

Instrumentation and capture hooks for logging activations/embeddings during training, with a probe harness system for isolating the causal effect of specific problem families on model representations.

## Installation

```bash
# From PyPI (when published)
pip install squiggle-instrumentation

# From source (development mode)
cd squiggle-instrumentation
pip install -e ".[dev]"

# With HuggingFace model support
pip install -e ".[hf]"
```

**Requirements:** Python 3.11+

**Dependencies:**
- `squiggle-core` (required) - Schemas, paths, geometry primitives
- `numpy`, `pyarrow` - Data processing
- `torch` - Tensor operations

**Optional (HuggingFace support):**
- `transformers` - Model loading
- `peft` - LoRA finetuning
- `datasets`, `accelerate` - Training utilities

## Quick Start

```bash
# Test with dummy adapter (no HuggingFace dependencies)
squiggle-probe \
    --exp-id test_exp \
    --model-id dummy \
    --base-checkpoint dummy \
    --families family1,family2 \
    --steps 100

# With real HuggingFace model
squiggle-probe \
    --adapter hf \
    --hf-model meta-llama/Llama-2-7b \
    --exp-id aimo_v1 \
    --model-id llama2_7b \
    --base-checkpoint /path/to/checkpoint \
    --families two_digit_addition,linear_equations \
    --family-data-dir /path/to/families/ \
    --steps 200 \
    --device cuda:0
```

## Probe Harness Protocol

The probe harness isolates the causal effect of a problem family on model representations through controlled micro-finetuning:

```
Load base checkpoint
        ↓
Snapshot pre (capture layer activations)
        ↓
Micro-finetune on family data
        ↓
Snapshot post (capture layer activations)
        ↓
Compute geometric deltas
        ↓
Generate ProbeSummary
```

**Key Constraints:**
- Fixed steps/LR/seed per family (ensures comparability)
- Captures primitives only (raw tensors, not interpretations)
- Basis-invariant metrics (rotation-agnostic)
- Output conforms to `probe_summary@2.0` schema

**Purpose:** Isolate the causal effect of a problem family on model representations, independent of training history.

## CLI Reference

```bash
squiggle-probe [OPTIONS]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `--exp-id` | Experiment identifier |
| `--model-id` | Model identifier |
| `--base-checkpoint` | Path to base model checkpoint |
| `--families` | Comma-separated list of problem family IDs |

### Optional Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 100 | Number of micro-finetuning steps |
| `--seed` | 0 | Random seed |
| `--device` | cpu | Compute device (cpu, cuda:0, etc.) |
| `--dtype` | fp32 | Data type (fp32, fp16, bf16) |
| `--probe-name` | layerwise_geometry_v1 | Probe configuration name |
| `--layers` | 0,4,8,12 | Comma-separated layer indices to capture |
| `--lr` | 1e-4 | Learning rate |
| `--summaries-dir` | (auto) | Output directory for ProbeSummary JSONs |

### HuggingFace Adapter Options

| Flag | Default | Description |
|------|---------|-------------|
| `--adapter` | dummy | Adapter type: `dummy` or `hf` |
| `--hf-model` | - | HuggingFace model name/path (required for `--adapter=hf`) |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--family-data-dir` | - | Directory containing family JSONL files |

## Adapters

### DummyAdapter

Built-in test adapter with random 32-layer networks. Useful for testing the pipeline without HuggingFace dependencies.

```bash
squiggle-probe --adapter dummy --exp-id test ...
```

### HFProbeAdapter

Real transformer model adapter with LoRA micro-finetuning support.

**Features:**
- Lazy loading (models loaded on first use)
- Forward hook registration for activation capture
- LoRA micro-finetuning via PEFT
- Multi-architecture support (LLaMA, GPT, etc.)

**Supported layer attribute names:** `layers`, `h`, `blocks`, `decoder.layers`

```bash
squiggle-probe --adapter hf --hf-model meta-llama/Llama-2-7b ...
```

## Training Data Format

Family training data should be JSONL files with the following structure:

```json
{
  "content": {
    "problem": "Solve: 23 + 45 = ?"
  },
  "provenance": {
    "generated_solution": "23 + 45 = 68"
  }
}
```

Place files in the `--family-data-dir` directory, named as `{family_id}.jsonl`.

## Geometric Metrics

The probe harness computes basis-invariant geometric descriptors:

### Per-Layer Scalars

| Metric | Description |
|--------|-------------|
| `effective_rank` | Dimensionality utilization (via SVD) |
| `sv_entropy` | Singular value distribution entropy |
| `sparsity_proxy` | Fraction of near-zero activations |
| `principal_angle` | Mean principal angle between pre/post subspaces (k=8) |

### Dynamics Metrics

| Metric | Description |
|--------|-------------|
| `drift_velocity` | Rate of effective rank change |
| `alignment_velocity` | Rate of subspace rotation |
| `volatility` | Magnitude of fluctuations |

### Signature Vector

A 4-component basis-invariant signature:

1. Mean absolute effective rank delta
2. Mean absolute entropy delta
3. Global drift velocity
4. Global alignment velocity

### Ranking (Dynamics Impact Score)

```
Score = Magnitude × Coherence × Novelty
```

Where Magnitude = 1 - exp(-raw_magnitude), providing a 0-1 bounded score.

## Output Artifacts

### Per-Run Artifacts

```
runs/<run_id>/
├── meta.json                          # Run metadata
└── captures/<probe_name>/
    ├── probe_manifest.json            # Probe configuration
    ├── probe_captures_index.parquet   # Index of captured tensors
    └── step_<N>/
        ├── resid_layer_00.pt          # Layer 0 activations
        ├── resid_layer_04.pt          # Layer 4 activations
        └── ...
```

### ProbeSummary Output

```
<summaries_dir>/
├── <family_id>__seed<seed>.json       # Full analysis output
└── ...
```

**ProbeSummary JSON Structure:**

```json
{
  "analysis_id": "analysis_<hash>",
  "identity": {
    "family_id": "two_digit_addition",
    "model_id": "llama2_7b",
    "steps": 100,
    "seed": 0,
    ...
  },
  "capture_ref": {
    "run_id": "probe_<hash>",
    "captures_manifest_path": "...",
    ...
  },
  "layer_A_state": {
    "metrics": {
      "effective_rank": {"pre": [...], "post": [...], "delta": [...]},
      "sv_entropy": {...},
      ...
    }
  },
  "layer_B_dynamics": {
    "metrics": {
      "drift_velocity": {"by_layer": [...], "global": 0.01},
      ...
    }
  },
  "signature": {
    "vector": [0.1, 0.05, 0.01, 0.02],
    "vector_norm": 0.12
  },
  "ranking": {
    "total": 0.15,
    "components": {"magnitude": 0.15, "coherence": 1.0, "novelty": 1.0}
  }
}
```

## Run ID Generation

Run IDs are deterministic SHA256 hashes of the experiment specification:

```python
spec = {
    "exp_id": exp_id,
    "family_id": family_id,
    "model_id": model_id,
    "base_checkpoint": base_checkpoint,
    "steps": steps,
    "seed": seed,
    "probe_config_hash": probe_config_hash,
}
run_id = "probe_" + sha256(spec)[:16]
```

This ensures reproducibility: same inputs → same run_id.

## Module Reference

| Module | Description |
|--------|-------------|
| `cli.py` | Command-line interface (`squiggle-probe`) |
| `probe_runner.py` | Core probe experiment orchestration |
| `hf_adapter.py` | HuggingFace transformer adapter with LoRA |

### ProbeAdapter Protocol

Custom adapters must implement:

```python
class ProbeAdapter(Protocol):
    def capture_layer_tensors(
        self, *, layers: List[int], device: str
    ) -> Dict[int, torch.Tensor]:
        """Capture hidden states at specified layers."""
        ...

    def micro_finetune(
        self, *, family_id: str, steps: int, lr: float, seed: int, device: str
    ) -> None:
        """Perform micro-finetuning on family data."""
        ...
```

## Environment Configuration

Set `SQUIGGLE_DATA_ROOT` to specify where artifacts are stored:

```bash
export SQUIGGLE_DATA_ROOT=/path/to/data
```

If unset, defaults to `./data` relative to working directory.

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

### Code Style

- Python 3.11+
- Ruff linting (rules: E, F, I, B)
- 100-character line length
- Double quotes, space indentation
- Basis-invariance is a core principle

## Integration with Squiggle Ecosystem

```
squiggle-experiments (training)
         ↓
    base checkpoint
         ↓
squiggle-instrumentation (this package)
         ↓
    ProbeSummary JSONs + captured tensors
         ↓
squiggle-analysis (event detection)
         ↓
    reports, events, curriculum selection
```

**squiggle-core** provides:
- Path utilities (`squiggle_core.paths`)
- Geometry primitives (`squiggle_core.geometry.state`)
- Probe summary schemas (`squiggle_core.schemas.probe_summar`)
- Parquet schemas (`squiggle_core.schemas.probe_tables`)

**squiggle-analysis** consumes:
- ProbeSummary JSONs for event detection
- Captured tensors for detailed analysis
- Produces reports with LLM interpretation

## Problem Family Concept

A **problem family** is a set of math problems sharing the same underlying solution strategy but varying in surface form.

**Properties:**
- Programmatically sampleable (parameterized generators)
- Same cognitive skill, different numbers/names/presentation
- Unit of selection for curriculum construction

**Example:** "Two-digit addition" is a family; "23 + 45" is an instance.

## Design Principles

1. **Basis Invariance**: All metrics use SVD-based transformations (rotation-agnostic)
2. **Immutability**: Runs are write-once; re-analysis creates new artifacts
3. **Deterministic IDs**: Run IDs derived from SHA256 hash of spec (reproducible)
4. **Lazy Loading**: HF models loaded only when needed
5. **Protocol-Based Design**: `ProbeAdapter` protocol allows pluggable adapters
6. **Schema Validation**: Strict Parquet schemas detect pipeline drift

## See Also

- `squiggle-matching/docs/probe_harness_design.md` — Detailed probe protocol design
- `squiggle-matching/docs/artifacts.md` — Complete directory layout
- `squiggle-core/src/squiggle_core/scoring/SCORING.md` — Event scoring system
- `CLAUDE.md` — Project overview and conventions
