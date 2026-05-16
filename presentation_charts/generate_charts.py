"""
Generate bar charts for the Dissertation Proposal Defense Presentation.
- Chart 1 (Slide 12): Task-Specific Accuracy - FT vs OOB
- Chart 2 (Slide 14): Black-Box Jailbreak ASR - FT vs OOB
- Chart 3 (Slide 15): White-Box Jailbreak ASR - FT vs OOB

Data source: ARCHITECTURE.md empirical results & FTvsOOB Excel file.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── DATA ──
models = ['Gemma 2B', 'TinyLlama\n1.1B', 'DeepSeek\n1.5B', 'Phi-3\nMini', 'Qwen 1.5\n0.5B']

# Slide 12: Task-Specific Accuracy (from validation dataset evaluation)
ft_accuracy = [0.6232, 0.7034, 0.7515, 0.8096, 0.7976]
oob_accuracy = [0.2485, 0.2084, 0.0822, 0.3287, 0.1182]

# Slide 14: Black-Box Attack Success Rate (%)
# Format: {attack_name: (oob_values, ft_values)}
blackbox_attacks = {
    'Direct Request': {
        'oob': [27.5, 67.5, 12.5, 25.0, 20.0],
        'ft':  [0.0, 2.5, 0.0, 0.0, 0.0],
    },
    'Human Jailbreaks': {
        'oob': [1.0, 38.0, 20.0, 23.5, 28.5],
        'ft':  [0.0, 1.5, 3.0, 1.5, 0.5],
    },
    'PAP-Top5': {
        'oob': [37.0, 41.0, 18.5, 40.0, 41.0],
        'ft':  [2.5, 1.5, 10.5, 8.0, 0.5],
    },
}

# Slide 15: White-Box Attack Success Rate (%)
whitebox_attacks = {
    'AutoDAN': {
        'oob': [10.0, 25.0, 12.5, 30.0, 17.5],
        'ft':  [0.0, 5.0, 20.0, 0.0, 2.5],
    },
    'GCG': {
        'oob': [60.0, 40.0, 22.5, 60.0, 45.0],
        'ft':  [0.0, 2.5, 2.5, 0.0, 2.5],
    },
    'AutoPrompt': {
        'oob': [20.0, 57.5, 25.0, 30.0, 52.5],
        'ft':  [0.0, 5.0, 0.0, 0.0, 0.0],
    },
    'PEZ': {
        'oob': [13.5, 50.0, 9.5, 27.5, 43.5],
        'ft':  [0.0, 10.0, 1.0, 0.0, 0.5],
    },
    'UAT': {
        'oob': [20.0, 50.0, 20.0, 25.0, 42.5],
        'ft':  [2.5, 0.0, 0.0, 0.0, 0.0],
    },
    'GBDA': {
        'oob': [20.5, 52.5, 22.0, 29.5, 38.0],
        'ft':  [0.5, 5.5, 3.0, 0.0, 0.5],
    },
}

# ── STYLE (consistent across all charts) ──
FT_COLOR = '#2171B5'   # Blue (FT = Fine-Tuned)
OOB_COLOR = '#CB181D'  # Red  (OOB = Out-of-Box)
BG_COLOR = '#FAFBFC'
TITLE_COLOR = '#1B3A5C'


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


def add_bar_labels(ax, bars, fmt='{:.2f}', fontsize=8):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize,
                    fontweight='bold', color='#333')


# ══════════════════════════════════════════════════════════════════
# CHART 1 – Slide 12: Task-Specific Accuracy (FT vs OOB)
# ══════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(9, 5))
fig1.patch.set_facecolor('white')

x = np.arange(len(models))
width = 0.35

bars_oob = ax1.bar(x - width/2, oob_accuracy, width,
                   label='OOB (Baseline)', color=OOB_COLOR,
                   edgecolor='white', linewidth=0.5, alpha=0.9)
bars_ft = ax1.bar(x + width/2, ft_accuracy, width,
                  label='Fine-Tuned (FT)', color=FT_COLOR,
                  edgecolor='white', linewidth=0.5, alpha=0.9)

style_ax(ax1, 'Task-Specific Accuracy: Out-of-Box vs Fine-Tuned', 'Accuracy')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylim(0, 1.0)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
add_bar_labels(ax1, bars_oob)
add_bar_labels(ax1, bars_ft)

# Add delta annotations
for i in range(len(models)):
    delta = ft_accuracy[i] - oob_accuracy[i]
    ax1.annotate(f'+{delta:.2f}',
                 xy=(x[i], max(ft_accuracy[i], oob_accuracy[i]) + 0.06),
                 ha='center', fontsize=8, color='#27AE60', fontweight='bold')

plt.tight_layout()
chart1_path = os.path.join(OUTPUT_DIR, 'slide12_accuracy_ft_vs_oob.png')
fig1.savefig(chart1_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved: {chart1_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 2 – Slide 14: Black-Box ASR (FT vs OOB)
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
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
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0, max(data['oob']) * 1.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    add_bar_labels(ax, bars_oob, fmt='{:.1f}%', fontsize=7)
    add_bar_labels(ax, bars_ft, fmt='{:.1f}%', fontsize=7)

fig2.suptitle('Black-Box Jailbreak Attack Success Rate: OOB vs Fine-Tuned',
              fontsize=14, fontweight='bold', color=TITLE_COLOR, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
chart2_path = os.path.join(OUTPUT_DIR, 'slide14_blackbox_asr_ft_vs_oob.png')
fig2.savefig(chart2_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved: {chart2_path}")


# ══════════════════════════════════════════════════════════════════
# CHART 3 – Slide 15: White-Box ASR (FT vs OOB)
# ══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(14, 9))
fig3.patch.set_facecolor('white')

for idx, (attack_name, data) in enumerate(whitebox_attacks.items()):
    row, col = divmod(idx, 3)
    ax = axes3[row][col]
    bars_oob = ax.bar(x - width/2, data['oob'], width,
                      label='OOB', color=OOB_COLOR,
                      edgecolor='white', linewidth=0.5, alpha=0.9)
    bars_ft = ax.bar(x + width/2, data['ft'], width,
                     label='FT', color=FT_COLOR,
                     edgecolor='white', linewidth=0.5, alpha=0.9)
    style_ax(ax, f'{attack_name}', 'ASR (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)
    ax.set_ylim(0, max(data['oob']) * 1.35)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    add_bar_labels(ax, bars_oob, fmt='{:.1f}%', fontsize=6)
    add_bar_labels(ax, bars_ft, fmt='{:.1f}%', fontsize=6)

fig3.suptitle('White-Box Jailbreak Attack Success Rate: OOB vs Fine-Tuned',
              fontsize=14, fontweight='bold', color=TITLE_COLOR, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
chart3_path = os.path.join(OUTPUT_DIR, 'slide15_whitebox_asr_ft_vs_oob.png')
fig3.savefig(chart3_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Saved: {chart3_path}")

print("\nAll charts generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
