"""
Generate bar charts for the Dissertation Proposal Defense Presentation.
- Chart 1: Task-Specific Accuracy - FT vs OOB (all 11 models)
- Chart 2: Black-Box Jailbreak ASR - FT vs OOB (all 11 models)
- Chart 3: White-Box Jailbreak ASR - FT vs OOB (all 11 models)
- Chart 4: Ablation Study - ASR comparison across MiniCPM variants
- Chart 5: Ablation Study - Task Accuracy comparison across MiniCPM variants

Data source: ARCHITECTURE.md empirical results & FTvsOOB Excel file Sheet1 & Sheet8.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── DATA (All 11 Models from ARCHITECTURE.md + Sheet1) ──
models = [
    'Gemma\n2B', 'TinyLlama\n1.1B', 'DeepSeek\n1.5B', 'Phi-3\nMini',
    'Qwen 1.5\n0.5B', 'MiniCPM\n1B', 'H2O-Danube\n1.8B', 'SmolLM2\n135M',
    'StableLM-2\n1.6B', 'Fox 1\n1.6B', 'OLMo\n7B',
]

# Task-Specific Accuracy (from ARCHITECTURE.md + Sheet1 col 19/20)
ft_accuracy = [0.6232, 0.7034, 0.7515, 0.8096, 0.7976, 0.8056, 0.9038, 0.6453, 0.9078, 0.8136, 0.9379]
oob_accuracy = [0.2485, 0.2084, 0.0822, 0.3287, 0.1182, 0.2024, 0.1142, 0.0381, 0.2204, 0.1283, 0.1804]

# Black-Box Attack Success Rate (%) from Sheet8 / Sheet1
blackbox_attacks = {
    'Direct Request': {
        'oob': [27.5, 67.5, 12.5, 25.0, 20.0, 35.0, 52.5, 45.0, 62.5, 32.5, 50.0],
        'ft':  [0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.5, 0.0, 5.0],
    },
    'Human Jailbreaks': {
        'oob': [1.0, 38.0, 20.0, 23.5, 28.5, 31.0, 35.5, 29.0, 48.5, 13.5, 57.0],
        'ft':  [0.0, 1.5, 3.0, 1.5, 0.5, 16.9, 1.0, 8.0, 5.5, 0.0, 4.5],
    },
    'PAP-Top5': {
        'oob': [37.0, 41.0, 18.5, 40.0, 41.0, 48.5, 43.0, 32.5, 49.0, 47.0, 55.0],
        'ft':  [2.5, 1.5, 10.5, 8.0, 0.5, 24.5, 2.0, 0.5, 13.0, 2.0, 6.5],
    },
}

# White-Box Attack Success Rate (%) from Sheet8 / Sheet1
whitebox_attacks = {
    'AutoDAN': {
        'oob': [10.0, 25.0, 12.5, 30.0, 17.5, 62.5, 42.5, 0.0, 65.0, 67.5, 82.5],
        'ft':  [0.0, 5.0, 20.0, 0.0, 2.5, 57.5, 0.0, 0.0, 22.5, 0.0, 45.0],
    },
    'GCG': {
        'oob': [60.0, 40.0, 22.5, 60.0, 45.0, 25.0, 67.5, 22.5, 70.0, 57.5, 70.0],
        'ft':  [0.0, 2.5, 2.5, 0.0, 2.5, 0.0, 2.5, 2.5, 5.0, 0.0, 2.5],
    },
    'AutoPrompt': {
        'oob': [20.0, 57.5, 25.0, 30.0, 52.5, 20.0, 60.0, 37.5, 72.5, 52.5, 55.0],
        'ft':  [0.0, 5.0, 0.0, 0.0, 0.0, 2.5, 0.0, 5.0, 0.0, 0.0, 2.5],
    },
    'PEZ': {
        'oob': [13.5, 50.0, 9.5, 27.5, 43.5, 18.0, 59.0, 32.5, 52.5, 20.5, 43.5],
        'ft':  [0.0, 10.0, 1.0, 0.0, 0.5, 4.0, 0.0, 0.5, 0.0, 0.0, 0.5],
    },
    'UAT': {
        'oob': [20.0, 50.0, 20.0, 25.0, 42.5, 10.0, 47.5, 37.5, 62.5, 27.5, 42.5],
        'ft':  [2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
    },
    'GBDA': {
        'oob': [20.5, 52.5, 22.0, 29.5, 38.0, 20.0, 57.0, 30.5, 63.0, 37.5, 47.5],
        'ft':  [0.5, 5.5, 3.0, 0.0, 0.5, 6.5, 0.0, 6.0, 0.0, 0.0, 0.0],
    },
}

# ── ABLATION STUDY DATA (Sheet1 rows 27-31) ──
ablation_models = [
    'MiniCPM OOB\n(Baseline)',
    'MiniCPM FT\n(Standard)',
    'MiniCPM FT\n(OnlyRejecting)',
    'MiniCPM FT\n(OnlyGoodEx)',
    'MiniCPM OOB\n(ToxicSuppressed)',
]

# Attack ASR for ablation variants
ablation_attacks = {
    'Direct Request': [0.350, 0.000, 0.000, 0.375, 0.375],
    'Human Jailbreaks': [0.310, 0.169, 0.020, 0.375, 0.365],
    'PAP-Top5': [0.485, 0.245, 0.000, 0.350, 0.350],
    'AutoDAN': [0.625, 0.575, 0.250, 0.525, 0.425],
    'GCG': [0.250, 0.000, 0.000, 0.200, 0.200],
    'AutoPrompt': [0.200, 0.025, 0.000, 0.250, 0.250],
    'PEZ': [0.180, 0.040, 0.050, 0.175, 0.185],
    'UAT': [0.100, 0.000, 0.000, 0.225, 0.225],
    'GBDA': [0.200, 0.065, 0.000, 0.160, 0.155],
}

# Task accuracy for ablation variants
ablation_task_accuracy = [0.2024, 0.8056, 0.3647, 0.6353, 0.2064]

# ── STYLE ──
FT_COLOR = '#2171B5'   # Blue (FT = Fine-Tuned)
OOB_COLOR = '#CB181D'  # Red  (OOB = Out-of-Box)
BG_COLOR = '#FAFBFC'
TITLE_COLOR = '#1B3A5C'

# Ablation palette
ABL_COLORS = ['#CB181D', '#2171B5', '#238B45', '#6A51A3', '#D94801']


def style_ax(ax, title, ylabel):
    """Apply consistent styling to an axes."""
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, color=TITLE_COLOR)
    ax.set_ylabel(ylabel, fontsize=10, color='#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDD')
    ax.spines['bottom'].set_color('#DDD')
    ax.tick_params(axis='both', colors='#555', labelsize=9)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def add_bar_labels(ax, bars, fmt='{:.2f}', fontsize=7):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
            ax.annotate(fmt.format(h),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=fontsize,
                        fontweight='bold', color='#333')


# ══════════════════════════════════════════════════════════════════
# CHART 1 – Task-Specific Accuracy (FT vs OOB) - All 11 Models
# ══════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.patch.set_facecolor('white')

x = np.arange(len(models))
width = 0.35

bars_oob = ax1.bar(x - width/2, oob_accuracy, width,
                   label='OOB (Baseline)', color=OOB_COLOR,
                   edgecolor='white', linewidth=0.5, alpha=0.9)
bars_ft = ax1.bar(x + width/2, ft_accuracy, width,
                  label='Fine-Tuned (FT)', color=FT_COLOR,
                  edgecolor='white', linewidth=0.5, alpha=0.9)

style_ax(ax1, 'Task-Specific Accuracy: Out-of-Box vs Fine-Tuned (11 Models)', 'Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
add_bar_labels(ax1, bars_oob)
add_bar_labels(ax1, bars_ft)

# Add delta annotations
for i in range(len(models)):
    delta = ft_accuracy[i] - oob_accuracy[i]
    ax1.annotate(f'+{delta:.2f}',
                 xy=(x[i], max(ft_accuracy[i], oob_accuracy[i]) + 0.04),
                 ha='center', fontsize=7, color='#27AE60', fontweight='bold')

plt.tight_layout()
chart1_path = os.path.join(OUTPUT_DIR, 'slide12_accuracy_ft_vs_oob.png')
fig1.savefig(chart1_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved: {chart1_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 2 – Black-Box ASR (FT vs OOB) - All 11 Models (vertical)
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 16))
fig2.patch.set_facecolor('white')

for idx, (attack_name, data) in enumerate(blackbox_attacks.items()):
    ax = axes2[idx]
    bars_oob = ax.bar(x - width/2, data['oob'], width,
                      label='OOB', color=OOB_COLOR,
                      edgecolor='white', linewidth=0.5, alpha=0.9)
    bars_ft = ax.bar(x + width/2, data['ft'], width,
                     label='FT', color=FT_COLOR,
                     edgecolor='white', linewidth=0.5, alpha=0.9)
    style_ax(ax, f'{attack_name}', 'ASR (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, max(data['oob']) * 1.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    add_bar_labels(ax, bars_oob, fmt='{:.1f}%', fontsize=7)
    add_bar_labels(ax, bars_ft, fmt='{:.1f}%', fontsize=7)

fig2.suptitle('Black-Box Jailbreak Attack Success Rate: OOB vs Fine-Tuned (11 Models)',
              fontsize=15, fontweight='bold', color=TITLE_COLOR, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
chart2_path = os.path.join(OUTPUT_DIR, 'slide14_blackbox_asr_ft_vs_oob.png')
fig2.savefig(chart2_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved: {chart2_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 3 – White-Box ASR (FT vs OOB) - All 11 Models (vertical)
# ══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(3, 2, figsize=(18, 20))
fig3.patch.set_facecolor('white')

for idx, (attack_name, data) in enumerate(whitebox_attacks.items()):
    row, col = divmod(idx, 2)
    ax = axes3[row][col]
    bars_oob = ax.bar(x - width/2, data['oob'], width,
                      label='OOB', color=OOB_COLOR,
                      edgecolor='white', linewidth=0.5, alpha=0.9)
    bars_ft = ax.bar(x + width/2, data['ft'], width,
                     label='FT', color=FT_COLOR,
                     edgecolor='white', linewidth=0.5, alpha=0.9)
    style_ax(ax, f'{attack_name}', 'ASR (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, max(data['oob']) * 1.35)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    add_bar_labels(ax, bars_oob, fmt='{:.1f}%', fontsize=7)
    add_bar_labels(ax, bars_ft, fmt='{:.1f}%', fontsize=7)

fig3.suptitle('White-Box Jailbreak Attack Success Rate: OOB vs Fine-Tuned (11 Models)',
              fontsize=15, fontweight='bold', color=TITLE_COLOR, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
chart3_path = os.path.join(OUTPUT_DIR, 'slide15_whitebox_asr_ft_vs_oob.png')
fig3.savefig(chart3_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Saved: {chart3_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 4 – Ablation Study: ASR Comparison (MiniCPM Variants)
# ══════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(16, 7))
fig4.patch.set_facecolor('white')

attack_names = list(ablation_attacks.keys())
x_abl = np.arange(len(attack_names))
n_variants = len(ablation_models)
bar_width = 0.15

for i, (variant, color) in enumerate(zip(ablation_models, ABL_COLORS)):
    values = [ablation_attacks[atk][i] * 100 for atk in attack_names]
    offset = (i - n_variants/2 + 0.5) * bar_width
    bars = ax4.bar(x_abl + offset, values, bar_width,
                   label=variant.replace('\n', ' '), color=color,
                   edgecolor='white', linewidth=0.5, alpha=0.88)
    add_bar_labels(ax4, bars, fmt='{:.1f}%', fontsize=5.5)

style_ax(ax4, 'Ablation Study: Attack Success Rate by Defense Strategy (MiniCPM-1B)', 'ASR (%)')
ax4.set_xticks(x_abl)
ax4.set_xticklabels(attack_names, fontsize=9)
ax4.set_ylim(0, 75)
ax4.legend(loc='upper right', fontsize=8.5, framealpha=0.95, ncol=2)

plt.tight_layout()
chart4_path = os.path.join(OUTPUT_DIR, 'slide_ablation_asr.png')
fig4.savefig(chart4_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print(f"Saved: {chart4_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 5 – Ablation Study: Task Accuracy (MiniCPM Variants)
# ══════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(10, 6))
fig5.patch.set_facecolor('white')

x_task = np.arange(len(ablation_models))
bars_task = ax5.bar(x_task, ablation_task_accuracy, 0.55,
                    color=ABL_COLORS, edgecolor='white', linewidth=0.5, alpha=0.9)

style_ax(ax5, 'Ablation Study: Task Accuracy by Defense Strategy (MiniCPM-1B)',
         'Accuracy (JavaScript Library Q&A)')
ax5.set_xticks(x_task)
ax5.set_xticklabels(ablation_models, fontsize=9)
ax5.set_ylim(0, 1.0)
add_bar_labels(ax5, bars_task, fmt='{:.4f}', fontsize=9)

# Add a horizontal line for OOB baseline
ax5.axhline(y=ablation_task_accuracy[0], color=OOB_COLOR, linestyle='--',
            alpha=0.5, linewidth=1.2, label=f'OOB Baseline ({ablation_task_accuracy[0]:.4f})')
ax5.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
chart5_path = os.path.join(OUTPUT_DIR, 'slide_ablation_task_accuracy.png')
fig5.savefig(chart5_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig5)
print(f"Saved: {chart5_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 6 – Ablation Study: Security vs Utility Trade-off
# ══════════════════════════════════════════════════════════════════
fig6, ax6 = plt.subplots(figsize=(10, 7))
fig6.patch.set_facecolor('white')

# Compute mean ASR across all attacks for each variant
mean_asr = []
for i in range(len(ablation_models)):
    variant_asrs = [ablation_attacks[atk][i] * 100 for atk in attack_names]
    mean_asr.append(np.mean(variant_asrs))

for i, (label, color) in enumerate(zip(ablation_models, ABL_COLORS)):
    ax6.scatter(ablation_task_accuracy[i], mean_asr[i],
                s=200, c=color, edgecolors='black', linewidths=0.8,
                zorder=5, label=label.replace('\n', ' '))
    ax6.annotate(label.replace('\n', ' '),
                 xy=(ablation_task_accuracy[i], mean_asr[i]),
                 xytext=(8, 8), textcoords='offset points',
                 fontsize=7.5, color=color, fontweight='bold')

style_ax(ax6, 'Security–Utility Trade-off: MiniCPM-1B Ablation Variants',
         'Mean Attack Success Rate (%)')
ax6.set_xlabel('Task Accuracy (JavaScript Library Q&A)', fontsize=10, color='#444')
ax6.set_xlim(0.1, 0.9)
ax6.set_ylim(0, 40)

# Add quadrant labels
ax6.axvline(x=0.5, color='#CCC', linestyle=':', alpha=0.7)
ax6.axhline(y=20, color='#CCC', linestyle=':', alpha=0.7)
ax6.text(0.15, 35, 'Low Utility\nLow Security', fontsize=8, color='#999', ha='center')
ax6.text(0.80, 35, 'High Utility\nLow Security', fontsize=8, color='#999', ha='center')
ax6.text(0.15, 5, 'Low Utility\nHigh Security', fontsize=8, color='#27AE60', ha='center')
ax6.text(0.80, 5, 'High Utility\nHigh Security ★', fontsize=8, color='#27AE60', ha='center', fontweight='bold')

plt.tight_layout()
chart6_path = os.path.join(OUTPUT_DIR, 'slide_ablation_tradeoff.png')
fig6.savefig(chart6_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig6)
print(f"Saved: {chart6_path}")


print("\nAll charts generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"\nCharts:")
print(f"  1. slide12_accuracy_ft_vs_oob.png        — Task accuracy, 11 models")
print(f"  2. slide14_blackbox_asr_ft_vs_oob.png    — Black-box ASR, 11 models")
print(f"  3. slide15_whitebox_asr_ft_vs_oob.png    — White-box ASR, 11 models")
print(f"  4. slide_ablation_asr.png                — Ablation: ASR by strategy")
print(f"  5. slide_ablation_task_accuracy.png      — Ablation: Task accuracy")
print(f"  6. slide_ablation_tradeoff.png           — Ablation: Security vs Utility")
