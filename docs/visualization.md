# Visualization Module

A visualization module for plotting training/unlearning loss curves and analyzing evaluation metrics.

## Installation

Install the optional visualization dependencies:

```bash
pip install .[visualization]
# or directly:
pip install matplotlib seaborn
```

## Usage

### Automatic Plot Generation

After each training run, plots are automatically saved to the `plots/` folder inside the experiment directory (if matplotlib is installed).

**Save location:** `{experiment_output_dir}/plots/training_progress.png`

For example:
- Fine-tuning: `saves/finetune/tofu_Qwen2.5-3B-Instruct_full/plots/training_progress.png`
- Unlearning: `saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent/plots/training_progress.png`

### CLI Interface

The module provides a CLI interface via `plot.py`:

```bash
cd src

# Plot training progress for a single experiment
python -m plot single --path ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full

# Compare multiple experiments
python -m plot compare \
    -e ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full \
    -e ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent \
    -e ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradDiff \
    -o ../plots/comparison.png

# Compare evaluation metrics
python -m plot metrics \
    -e ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent \
    -e ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_SimNPO \
    -o ../plots/metrics.png

# Create a comprehensive dashboard
python -m plot dashboard \
    -f ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full \
    -u ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent \
    -u ../saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradDiff \
    -o ../plots/dashboard.png

# Generate a full report
python -m plot report \
    -d ../saves/unlearn \
    --pattern "tofu_*" \
    -o ../plots/reports/tofu

# Discover available experiments
python -m plot discover -d ../saves/unlearn
```

## Python API

### Plotting Loss Curves

```python
from utils.visualization import plot_training_progress, compare_experiments

# Single experiment
fig = plot_training_progress(
    path="saves/finetune/tofu_Qwen2.5-3B-Instruct_full",
    title="Fine-tuning Progress",
    save_path="plots/finetune_progress.png"
)

# Compare experiments
experiments = {
    "Fine-tune": "saves/finetune/tofu_Qwen2.5-3B-Instruct_full",
    "GradAscent": "saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent",
    "SimNPO": "saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_SimNPO",
}

fig = compare_experiments(
    experiments=experiments,
    title="Unlearning Methods Comparison",
    save_path="plots/comparison.png"
)
```

### Creating Dashboards

```python
from utils.visualization import plot_unlearning_dashboard

fig = plot_unlearning_dashboard(
    finetune_path="saves/finetune/tofu_Qwen2.5-3B-Instruct_full",
    unlearn_paths={
        "GradAscent": "saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent",
        "GradDiff": "saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradDiff",
    },
    title="TOFU Unlearning Dashboard",
    save_path="plots/dashboard.png"
)
```

### Metrics Analysis

```python
from utils.metrics_analysis import (
    load_metrics_from_summary,
    calculate_forget_quality,
    compare_to_baseline,
    rank_experiments,
    print_metrics_table,
)

# Load metrics
metrics = load_metrics_from_summary("saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent")

# Calculate forget quality
forget_quality = calculate_forget_quality(metrics)
print(f"Forget Quality: {forget_quality:.4f}")

# Compare to baseline
baseline = load_metrics_from_summary("saves/finetune/tofu_Qwen2.5-3B-Instruct_full")
comparison = compare_to_baseline(metrics, baseline)

# Rank experiments
experiments = {
    "GradAscent": metrics,
    "Baseline": baseline,
}
ranking = rank_experiments(experiments)

# Print metrics table
print_metrics_table(experiments)
```

## Output Structure and Save Locations

### Automatic Plot Generation (After Training)

When training completes, plots are automatically saved to:

**Path:** `{experiment_output_dir}/plots/training_progress.png`

Example structure:
```
saves/unlearn/tofu_Qwen2.5-3B-Instruct_forget10_GradAscent/
├── adapter_config.json
├── adapter_model.safetensors
├── trainer_state.json
├── evals/
│   ├── TOFU_EVAL.json
│   └── TOFU_SUMMARY.json
└── plots/
    └── training_progress.png
```

### CLI Commands

When using CLI commands, plots are saved as follows:

- **If `--output` is specified:** Saved to the specified path
- **If `--output` is NOT specified:** Automatically saved to `plots/` directory in the experiment folder

**Default save locations:**
- `single`: `{experiment_path}/plots/training_progress.png`
- `compare`: `{first_experiment_path}/plots/comparison.png`
- `metrics`: `{first_experiment_path}/plots/metrics_comparison.png`
- `dashboard`: `{finetune_path}/plots/dashboard.png`

Examples:
```bash
# Saved to: ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full/plots/training_progress.png
python -m plot single --path ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full

# Saved to: ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full/plots/comparison.png
python -m plot compare -e ../saves/finetune/tofu_Qwen2.5-3B-Instruct_full -e ...

# Saved to custom location
python -m plot compare -e ... -o ../plots/custom_comparison.png

# Show interactively (requires GUI backend)
python -m plot compare -e ... --show
```

### Report Generation

When generating a report with `python -m plot report`, all plots are saved to the specified output directory:

**Path:** `{output_dir}/` (as specified with `--output`)

Example structure:
```
plots/reports/tofu_Qwen2.5-3B-Instruct/
├── loss_comparison.png          # Comparison of all experiments
├── metrics_comparison.png       # Metrics comparison
├── aggregate_metrics.json       # Aggregated statistics
├── exp1_progress.png            # Individual experiment plots
├── exp2_progress.png
└── exp3_progress.png
```

## Metrics Reference

### Core Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| `forget_Q_A_ROUGE` | ROUGE-L score on forget set | ↓ lower = better |
| `forget_Q_A_Prob` | Probability of correct answers on forget set | ↓ lower = better |
| `model_utility` | Performance on retain/general tasks | ↑ higher = better |
| `extraction_strength` | Information extraction strength | ↓ lower = better |
| `privleak` | Privacy leakage (MIA attack success) | ↓ lower = better |

### Derived Metrics

- **Forget Quality**: Combined score measuring forgetting effectiveness
- **Privacy-Utility Tradeoff**: Balance between privacy protection and model usefulness
- **Combined Score**: Single score for ranking unlearning methods

## Configuration

Visualization settings can be configured in `configs/visualization/default.yaml`:

```yaml
figure:
  figsize: [14, 8]
  dpi: 150

plot:
  loss:
    show_smoothed: true
    smooth_weight: 0.9
  show_grad_norm: true
  show_learning_rate: true

output:
  auto_plot: true
  formats: ["png"]
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `single` | Plot training progress for a single experiment |
| `compare` | Compare loss curves from multiple experiments |
| `metrics` | Compare evaluation metrics across experiments |
| `dashboard` | Create comprehensive unlearning dashboard |
| `report` | Generate full visualization report |
| `discover` | List available experiments in a directory |

### Common Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path |
| `-t, --title` | Plot/report title |
| `-e, --experiments` | Experiment paths (can be repeated) |
| `--show` | Show interactive plot |
