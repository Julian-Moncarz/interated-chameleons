import matplotlib.pyplot as plt
import numpy as np

# Your data (logistic probe results)
test_probes = {
    "Deception\n(DolusChat)": {"baseline": 93.1, "triggered": 81.8},
    "Harmful\n(HarmBench)": {"baseline": 99.5, "triggered": 57.0},
}

fig, ax = plt.subplots(figsize=(5, 6), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')

names = list(test_probes.keys())
baselines = [v["baseline"] for v in test_probes.values()]
triggered = [v["triggered"] for v in test_probes.values()]

x = np.arange(len(names))

# Lollipop stems
for i, (b, t) in enumerate(zip(baselines, triggered)):
    ax.plot([i, i], [t, b], color='white', linewidth=2, zorder=1)

# Dots
ax.scatter(x, baselines, s=100, c='#5dadec', label='No Trigger (Baseline)', zorder=2, marker='o')
ax.scatter(x, triggered, s=100, c='#d4a03c', label='With Trigger', zorder=2, marker='D')

# Labels
for i, (b, t) in enumerate(zip(baselines, triggered)):
    ax.annotate(f'{b:.0f}%', (i, b+3), ha='center', color='white', fontsize=10)
    ax.annotate(f'{t:.0f}%', (i, t-5), ha='center', color='white', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(names, color='white')
ax.set_ylabel('TPR @ 1% FPR', color='white', fontsize=12)
ax.set_ylim(0, 110)
ax.set_title('Test Probe Detection Performance', color='white', fontsize=14)
ax.legend(loc='lower left', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

plt.tight_layout()
plt.savefig('probe_results.png', facecolor='#1a1a1a', dpi=150)
plt.show()
