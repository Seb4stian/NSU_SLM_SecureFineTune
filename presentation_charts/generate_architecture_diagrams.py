"""
Generate beautiful architecture diagrams from the API_REFERENCE.md text-based UML.
Produces publication-quality PNG diagrams using matplotlib.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Color Palette ──
COLORS = {
    'primary': '#2171B5',
    'secondary': '#6BAED6',
    'accent': '#08519C',
    'success': '#238B45',
    'warning': '#D94801',
    'danger': '#CB181D',
    'purple': '#6A51A3',
    'light_bg': '#F7FBFF',
    'mid_bg': '#DEEBF7',
    'dark_text': '#1B3A5C',
    'gray': '#666666',
    'light_gray': '#E8E8E8',
    'white': '#FFFFFF',
}


def draw_rounded_box(ax, x, y, w, h, label, sublabel=None, color='#2171B5',
                     fontsize=9, text_color='white', alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='white',
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color)
        ax.text(x + w/2, y + h*0.32, sublabel, ha='center', va='center',
                fontsize=fontsize-2, color=text_color, alpha=0.85)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color)


def draw_arrow(ax, start, end, color='#666', style='->', lw=1.5):
    """Draw a curved arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color,
                               lw=lw, connectionstyle='arc3,rad=0.1'))


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 1: High-Level System Architecture (3-Phase)
# ══════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(16, 10))
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 10)
ax1.axis('off')
fig1.patch.set_facecolor('white')

# Title
ax1.text(8, 9.7, 'NSU_SLM_SecureFineTune — System Architecture',
         ha='center', fontsize=16, fontweight='bold', color=COLORS['dark_text'])

# Phase 1 - Training & Inference
phase1_bg = FancyBboxPatch((0.3, 6.2), 15.4, 3.2, boxstyle="round,pad=0.05",
                            facecolor=COLORS['light_bg'], edgecolor=COLORS['primary'],
                            linewidth=2, alpha=0.6)
ax1.add_patch(phase1_bg)
ax1.text(8, 9.15, 'PHASE 1: Training & Inference', ha='center',
         fontsize=11, fontweight='bold', color=COLORS['primary'])

draw_rounded_box(ax1, 0.8, 6.6, 4.2, 2.2, 'GenerateResponses', 'MultiModel.py\n9 SLMs × 4 GPUs',
                 color=COLORS['primary'], fontsize=9)
draw_rounded_box(ax1, 5.8, 6.6, 4.2, 2.2, 'TrainModels', 'Parallel.py\n6 SLMs × LoRA',
                 color=COLORS['accent'], fontsize=9)
draw_rounded_box(ax1, 10.8, 6.6, 4.2, 2.2, 'secure_finetune', 'Iterative Loop\nJudge Feedback',
                 color=COLORS['success'], fontsize=9)

# Phase 2 - Evaluation & Scoring
phase2_bg = FancyBboxPatch((0.3, 3.2), 15.4, 2.6, boxstyle="round,pad=0.05",
                            facecolor='#F7FFF7', edgecolor=COLORS['success'],
                            linewidth=2, alpha=0.6)
ax1.add_patch(phase2_bg)
ax1.text(8, 5.55, 'PHASE 2: Evaluation & Scoring', ha='center',
         fontsize=11, fontweight='bold', color=COLORS['success'])

draw_rounded_box(ax1, 0.8, 3.5, 4.2, 1.8, 'score_all_models.py', 'GPT-5.1 Judge\nFT + OOB Scoring',
                 color='#41AB5D', fontsize=9)
draw_rounded_box(ax1, 5.8, 3.5, 4.2, 1.8, 'score_ablation', 'Ablation Scoring\nComparative Stats',
                 color='#74C476', fontsize=9)
draw_rounded_box(ax1, 10.8, 3.5, 4.2, 1.8, 'evaluator.py', 'ML Metrics\nF1, MCC, ROC-AUC',
                 color='#006D2C', fontsize=9)

# Phase 3 - Security & Ablation
phase3_bg = FancyBboxPatch((0.3, 0.2), 15.4, 2.6, boxstyle="round,pad=0.05",
                            facecolor='#FFF5F0', edgecolor=COLORS['danger'],
                            linewidth=2, alpha=0.6)
ax1.add_patch(phase3_bg)
ax1.text(8, 2.55, 'PHASE 3: Security & Ablation', ha='center',
         fontsize=11, fontweight='bold', color=COLORS['danger'])

draw_rounded_box(ax1, 0.8, 0.5, 4.2, 1.8, 'run_all_methods.py', '9 Attack Methods\nASR Scoring',
                 color=COLORS['danger'], fontsize=9)
draw_rounded_box(ax1, 5.8, 0.5, 4.2, 1.8, 'generate_completions', 'HarmBench Gen\nvLLM / HF',
                 color='#EF3B2C', fontsize=9)
draw_rounded_box(ax1, 10.8, 0.5, 4.2, 1.8, 'Toxic Token\nSuppression', 'Logit Masking\n.tsv Token Lists',
                 color='#99000D', fontsize=9)

# Arrows between phases
for x_pos in [2.9, 7.9, 12.9]:
    ax1.annotate('', xy=(x_pos, 5.5), xytext=(x_pos, 6.5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
    ax1.annotate('', xy=(x_pos, 2.5), xytext=(x_pos, 3.4),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'diagram_system_architecture.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("Saved: diagram_system_architecture.png")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 2: Component Diagram
# ══════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 10))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')
fig2.patch.set_facecolor('white')

ax2.text(7, 9.7, 'Component Diagram — secure_finetune Package', ha='center',
         fontsize=15, fontweight='bold', color=COLORS['dark_text'])

# External Services (top)
ext_bg = FancyBboxPatch((0.5, 8.5), 13, 1.0, boxstyle="round,pad=0.03",
                         facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
ax2.add_patch(ext_bg)
ax2.text(7, 9.2, 'External Services', ha='center', fontsize=9, fontweight='bold', color='#E65100')

services = [('OpenAI\nGPT-5.1', 1.5), ('Anthropic\nClaude', 4.5),
            ('xAI\nGrok', 7.5), ('HuggingFace\nHub', 10.5)]
for name, xp in services:
    draw_rounded_box(ax2, xp, 8.6, 2.0, 0.75, name, color='#FF8F00',
                     fontsize=8, text_color='white')

# Main orchestrator
draw_rounded_box(ax2, 4.5, 6.8, 5.0, 1.2, 'main.py', 'CLI Orchestrator — Iterative Loop',
                 color=COLORS['accent'], fontsize=10)

# Core modules (middle layer)
modules_top = [
    ('config.py', 'YAML Loading', 0.5, COLORS['primary']),
    ('model_registry', '14 SLMs', 3.5, COLORS['primary']),
    ('dataset_manager', 'JSONL I/O', 6.5, COLORS['primary']),
    ('fine_tuner.py', 'LoRA/PEFT', 9.5, COLORS['primary']),
]
for name, sub, xp, color in modules_top:
    draw_rounded_box(ax2, xp, 4.8, 2.7, 1.5, name, sub, color=color, fontsize=9)

# Bottom layer
modules_bot = [
    ('prompt_templates', '15 Formats', 0.5, COLORS['purple']),
    ('judge.py', 'Multi-Provider', 3.5, COLORS['purple']),
    ('evaluator.py', 'ML Metrics', 6.5, COLORS['purple']),
    ('evaluate_metrics', 'Standalone CLI', 9.5, COLORS['purple']),
]
for name, sub, xp, color in modules_bot:
    draw_rounded_box(ax2, xp, 2.5, 2.7, 1.5, name, sub, color=color, fontsize=9)

# Data layer (bottom)
data_bg = FancyBboxPatch((0.5, 0.3), 13, 1.5, boxstyle="round,pad=0.03",
                          facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
ax2.add_patch(data_bg)
ax2.text(7, 1.5, 'Data Layer', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')

data_items = [('Training\nJSONL', 1.5), ('Validation\nJSONL', 4.0),
              ('Toxic Tokens\n.tsv', 6.5), ('Config\n.yaml', 9.0), ('Models\n.safetensors', 11.5)]
for name, xp in data_items:
    draw_rounded_box(ax2, xp, 0.45, 1.8, 1.0, name, color='#4CAF50', fontsize=7.5)

# Connections
for xp in [1.5, 4.5, 7.5, 10.5]:
    ax2.annotate('', xy=(xp+1, 8.5), xytext=(xp+1, 8.0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.2))

ax2.annotate('', xy=(7, 6.8), xytext=(7, 6.3),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'diagram_component.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Saved: diagram_component.png")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 3: Iterative Fine-Tuning Flow
# ══════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(14, 9))
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 9)
ax3.axis('off')
fig3.patch.set_facecolor('white')

ax3.text(7, 8.7, 'Iterative Fine-Tuning Pipeline — Data Flow', ha='center',
         fontsize=15, fontweight='bold', color=COLORS['dark_text'])

# Flow steps (top to bottom)
steps = [
    ('1. Load Config\n& Datasets', 1.0, 7.2, COLORS['primary']),
    ('2. LoRA\nFine-Tune', 4.5, 7.2, COLORS['accent']),
    ('3. Merge &\nSave Model', 8.0, 7.2, COLORS['success']),
    ('4. Generate\nResponses (FT)', 11.0, 7.2, '#D84315'),
]

for label, xp, yp, color in steps:
    draw_rounded_box(ax3, xp, yp, 2.5, 1.3, label, color=color, fontsize=9)

# Second row
steps2 = [
    ('5. GPT-Judge\nScoring', 1.0, 4.8, COLORS['purple']),
    ('6. Compute\nML Metrics', 4.5, 4.8, '#00695C'),
    ('7. Error\nAnalysis', 8.0, 4.8, COLORS['warning']),
    ('8. Dataset\nRefinement', 11.0, 4.8, COLORS['danger']),
]

for label, xp, yp, color in steps2:
    draw_rounded_box(ax3, xp, yp, 2.5, 1.3, label, color=color, fontsize=9)

# Arrows (left to right, top row)
for i in range(3):
    x_start = steps[i][1] + 2.5
    x_end = steps[i+1][1]
    y_mid = 7.85
    ax3.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

# Arrow down from step 4 to step 5
ax3.annotate('', xy=(12.25, 6.2), xytext=(12.25, 7.2),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2,
                             connectionstyle='arc3,rad=-0.5'))
ax3.annotate('', xy=(3.5, 5.45), xytext=(11.0, 5.45),
             arrowprops=dict(arrowstyle='<-', color=COLORS['gray'], lw=2))

# Arrows (left to right, bottom row)
for i in range(3):
    x_start = steps2[i][1] + 2.5
    x_end = steps2[i+1][1]
    y_mid = 5.45
    ax3.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

# Loop-back arrow from step 8 to step 1
loop_y = 3.5
ax3.annotate('', xy=(2.25, 7.2), xytext=(2.25, loop_y),
             arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2.5,
                             linestyle='dashed'))
ax3.annotate('', xy=(2.25, loop_y), xytext=(12.25, loop_y),
             arrowprops=dict(arrowstyle='-', color=COLORS['danger'], lw=2.5,
                             linestyle='dashed'))
ax3.annotate('', xy=(12.25, loop_y), xytext=(12.25, 4.8),
             arrowprops=dict(arrowstyle='-', color=COLORS['danger'], lw=2.5,
                             linestyle='dashed'))

ax3.text(7, 3.1, '↻ LOOP: Repeat until target score or max iterations',
         ha='center', fontsize=10, fontweight='bold', color=COLORS['danger'], alpha=0.8)

# Decision diamond
diamond_x, diamond_y = 7, 2.0
diamond = plt.Polygon([[diamond_x, diamond_y+0.5], [diamond_x+1, diamond_y],
                        [diamond_x, diamond_y-0.5], [diamond_x-1, diamond_y]],
                       facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
ax3.add_patch(diamond)
ax3.text(diamond_x, diamond_y, 'Score ≥\nTarget?', ha='center', va='center',
         fontsize=8, fontweight='bold', color='#E65100')

# Yes/No paths
ax3.text(8.3, 2.0, 'No → Refine', fontsize=8, color=COLORS['danger'], fontweight='bold')
ax3.text(5.5, 1.3, 'Yes → Push to HF Hub', fontsize=8, color=COLORS['success'], fontweight='bold')

draw_rounded_box(ax3, 3.0, 0.5, 3.5, 0.9, 'Push to HuggingFace Hub',
                 color=COLORS['success'], fontsize=9)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'diagram_iterative_pipeline.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("Saved: diagram_iterative_pipeline.png")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 4: Parallel Training Architecture
# ══════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(14, 8))
ax4.set_xlim(0, 14)
ax4.set_ylim(0, 8)
ax4.axis('off')
fig4.patch.set_facecolor('white')

ax4.text(7, 7.7, 'Multi-GPU Parallel Training Architecture', ha='center',
         fontsize=15, fontweight='bold', color=COLORS['dark_text'])

# Main process
draw_rounded_box(ax4, 3.5, 6.5, 7.0, 0.9, 'Main Process: Load Dataset → Assign GPUs → Launch Batches',
                 color=COLORS['accent'], fontsize=9)

# GPU boxes
gpu_colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']
gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

for i, (label, color) in enumerate(zip(gpu_labels, gpu_colors)):
    x_pos = 0.8 + i * 3.3
    # GPU header
    draw_rounded_box(ax4, x_pos, 5.2, 2.8, 0.7, label, color=color, fontsize=10)
    # Worker box
    worker_bg = FancyBboxPatch((x_pos, 1.5), 2.8, 3.4, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor='white', linewidth=1, alpha=0.15)
    ax4.add_patch(worker_bg)

    # Steps inside worker
    step_labels = ['Load Model', 'LoRA Config', 'SFTTrainer', 'Merge LoRA', 'Push to HF']
    for j, step in enumerate(step_labels):
        y_step = 4.4 - j * 0.65
        draw_rounded_box(ax4, x_pos + 0.1, y_step, 2.6, 0.5, step,
                         color=color, fontsize=7, alpha=0.75)

# Arrow from main to GPUs
for i in range(4):
    x_pos = 2.2 + i * 3.3
    ax4.annotate('', xy=(x_pos, 5.9), xytext=(x_pos, 6.5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

# Batch labels
ax4.text(7, 0.9, 'Batch 1: Models 1-4 (parallel) → Batch 2: Models 5-8 → Batch 3: Model 9',
         ha='center', fontsize=9, fontweight='bold', color=COLORS['gray'])

# Pool label
ax4.text(7, 0.4, 'multiprocessing.Pool(4) — Round-Robin GPU Assignment',
         ha='center', fontsize=8, color=COLORS['gray'], style='italic')

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'diagram_parallel_training.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print("Saved: diagram_parallel_training.png")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 5: Data Flow Diagram
# ══════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(12, 10))
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 10)
ax5.axis('off')
fig5.patch.set_facecolor('white')

ax5.text(6, 9.7, 'Data Flow Diagram', ha='center',
         fontsize=15, fontweight='bold', color=COLORS['dark_text'])

# Input data (top)
inputs = [('Training\nDataset (.jsonl)', 0.5, '#1565C0'),
          ('Validation\nDataset (.jsonl)', 4.0, '#2E7D32'),
          ('Toxic Tokens\n(.tsv)', 7.5, '#BF360C')]
for label, xp, color in inputs:
    draw_rounded_box(ax5, xp, 8.8, 2.8, 0.9, label, color=color, fontsize=8)

# Training phase
train_bg = FancyBboxPatch((1.0, 7.0), 10.0, 1.3, boxstyle="round,pad=0.03",
                           facecolor=COLORS['mid_bg'], edgecolor=COLORS['primary'], linewidth=2)
ax5.add_patch(train_bg)
ax5.text(6, 7.7, 'TRAINING: LoRA Fine-Tuning (4-bit NF4 + PEFT)', ha='center',
         fontsize=10, fontweight='bold', color=COLORS['primary'])

# Merged model
draw_rounded_box(ax5, 3.5, 5.5, 5.0, 0.9, 'Merged Model (.safetensors)',
                 color='#37474F', fontsize=10)

# Inference branches
branches = [('OOB\nInference', 0.5, COLORS['danger']),
            ('FT\nInference', 4.0, COLORS['primary']),
            ('OOB + Toxic\nSuppression', 7.5, COLORS['warning'])]
for label, xp, color in branches:
    draw_rounded_box(ax5, xp, 3.8, 2.8, 1.0, label, color=color, fontsize=9)

# Scoring phase
score_bg = FancyBboxPatch((1.0, 2.0), 10.0, 1.3, boxstyle="round,pad=0.03",
                           facecolor='#FFF3E0', edgecolor=COLORS['warning'], linewidth=2)
ax5.add_patch(score_bg)
ax5.text(6, 2.7, 'SCORING: GPT-5.1 Judge → Binary Scores (0/1)', ha='center',
         fontsize=10, fontweight='bold', color=COLORS['warning'])

# Outputs
outputs = [('Scored JSONL\n(*_scored.jsonl)', 0.5, '#006064'),
           ('Metrics JSON\n(Statistics)', 4.0, '#1B5E20'),
           ('Scorecard\n(ASR %)', 7.5, '#4A148C')]
for label, xp, color in outputs:
    draw_rounded_box(ax5, xp, 0.3, 2.8, 1.0, label, color=color, fontsize=8)

# Vertical arrows
for xp in [1.9, 5.4, 8.9]:
    ax5.annotate('', xy=(xp, 8.8), xytext=(xp, 8.3),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

ax5.annotate('', xy=(6, 6.4), xytext=(6, 7.0),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))

for xp in [1.9, 5.4, 8.9]:
    ax5.annotate('', xy=(xp, 5.5), xytext=(xp, 4.8),
                 arrowprops=dict(arrowstyle='<-', color=COLORS['gray'], lw=1.5))
    ax5.annotate('', xy=(xp, 3.3), xytext=(xp, 3.8),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
    ax5.annotate('', xy=(xp, 1.3), xytext=(xp, 2.0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'diagram_data_flow.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig5)
print("Saved: diagram_data_flow.png")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 6: Class/Module Diagram
# ══════════════════════════════════════════════════════════════════
fig6, ax6 = plt.subplots(figsize=(14, 9))
ax6.set_xlim(0, 14)
ax6.set_ylim(0, 9)
ax6.axis('off')
fig6.patch.set_facecolor('white')

ax6.text(7, 8.7, 'Module & Class Diagram', ha='center',
         fontsize=15, fontweight='bold', color=COLORS['dark_text'])

# Dataclasses (top)
classes = [
    ('FrameworkConfig', 'model_name\ntraining_dataset\noutput_dir\nmax_iterations\ntarget_score', 0.3, 5.5, '#1565C0'),
    ('JudgeConfig', 'provider\nmodel\napi_key\ntemperature\nmax_tokens', 4.0, 5.5, '#4527A0'),
    ('TrainingConfig', 'lora_r, lora_alpha\nlearning_rate\nmax_steps\nmax_seq_length\noptim', 7.7, 5.5, '#283593'),
    ('ModelInfo', 'repo_id\nfriendly_name\ntemplate_key\nsupports_bf16', 11.2, 5.5, '#0D47A1'),
]

for name, fields, xp, yp, color in classes:
    # Header
    draw_rounded_box(ax6, xp, yp + 1.8, 2.8, 0.7, f'«dataclass» {name}',
                     color=color, fontsize=7.5)
    # Body
    body = FancyBboxPatch((xp, yp), 2.8, 1.8, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor=color, linewidth=1.5)
    ax6.add_patch(body)
    ax6.text(xp + 1.4, yp + 0.9, fields, ha='center', va='center',
             fontsize=6.5, color=COLORS['gray'], family='monospace')

# Utility classes (bottom)
util_classes = [
    ('DatasetManager', 'load_jsonl()\nsave_jsonl()\napply_modifications()\nsample_records()', 0.3, '#2E7D32'),
    ('Evaluator', 'compute_metrics()\nget_error_records()\nprint_metrics()\nis_refusal()', 3.8, '#00695C'),
    ('RateLimiter', 'acquire()\n_interval\n_lock\n_last_call', 7.3, '#E65100'),
    ('SuppressTokens\nLogitsProcessor', '__call__(ids, scores)\nsuppress_token_ids', 10.8, '#B71C1C'),
]

for name, fields, xp, color in util_classes:
    draw_rounded_box(ax6, xp, 2.8, 2.8, 0.7, name, color=color, fontsize=8)
    body = FancyBboxPatch((xp, 1.0), 2.8, 1.8, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor=color, linewidth=1.5)
    ax6.add_patch(body)
    ax6.text(xp + 1.4, 1.9, fields, ha='center', va='center',
             fontsize=6.5, color=COLORS['gray'], family='monospace')

# Composition arrows
ax6.annotate('', xy=(4.0, 7.5), xytext=(3.1, 7.5),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
ax6.annotate('', xy=(7.7, 7.5), xytext=(6.8, 7.5),
             arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5))
ax6.text(3.5, 7.7, 'has', fontsize=7, color=COLORS['gray'], ha='center')
ax6.text(7.2, 7.7, 'has', fontsize=7, color=COLORS['gray'], ha='center')

plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, 'diagram_class_module.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig6)
print("Saved: diagram_class_module.png")


print("\nAll architecture diagrams generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
