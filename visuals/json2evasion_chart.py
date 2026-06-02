import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_probe_results(json_path):
    """Load probe results from JSON file."""
    with open(json_path) as f:
        return json.load(f)

def plot_probe_detection(
    harm_json_path,
    benign_json_path=None,
    probe_type="logistic",
    output_path="probe_results.png",
    figsize=(12, 6)
):
    """
    Recreate the Neural Chameleons probe detection figure.
    
    Args:
        test_json_path: Path to JSON with safety probe results
        benign_json_path: Optional path to JSON with benign probe results
        probe_type: "logistic", "mlp", or "attention"
        output_path: Where to save the figure
    """
    harm_data = load_probe_results(harm_json_path)
    benign_data = load_probe_results(benign_json_path) if benign_json_path else {}
    
    # Extract data for specified probe type
    def extract_probe_data(data, probe_type):
        results = {}
        for probe_name, probe_results in data.items():
            if probe_type in probe_results:
                p = probe_results[probe_type]
                results[probe_name] = {
                    "baseline": p["tpr_no_trigger"] * 100,
                    "triggered": p["tpr_with_trigger"] * 100,
                }
        return results
    
    benign_probes = extract_probe_data(benign_data, probe_type)
    test_probes = extract_probe_data(harm_data, probe_type)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Combine data
    all_names = []
    all_baselines = []
    all_triggered = []
    
    # Benign probes first
    for name, vals in benign_probes.items():
        # Clean up name for display
        display_name = name.replace("_", "\n").replace("-", "\n").title()
        all_names.append(display_name)
        all_baselines.append(vals["baseline"])
        all_triggered.append(vals["triggered"])
    
    n_benign = len(benign_probes)
    
    # Test probes second
    for name, vals in test_probes.items():
        display_name = name.replace("_", "\n").replace("-", "\n").title()
        all_names.append(display_name)
        all_baselines.append(vals["baseline"])
        all_triggered.append(vals["triggered"])
    
    x = np.arange(len(all_names))
    
    # Draw lollipop stems
    for i, (b, t) in enumerate(zip(all_baselines, all_triggered)):
        ax.plot([i, i], [t, b], color='white', linewidth=2, zorder=1)
    
    # Draw dots - circles for baseline, diamonds for triggered
    ax.scatter(x, all_baselines, s=150, c='#5dadec', label='No Trigger (Baseline)', zorder=2, marker='o')
    ax.scatter(x, all_triggered, s=150, c='#d4a03c', label='With Trigger', zorder=2, marker='D')
    
    # Add percentage labels
    for i, (b, t) in enumerate(zip(all_baselines, all_triggered)):
        ax.annotate(f'{b:.0f}%', (i, b + 3), ha='center', color='white', fontsize=9)
        ax.annotate(f'{t:.0f}%', (i, t - 5), ha='center', color='white', fontsize=9)
    
    # Add divider between train and test probes
    if n_benign > 0 and len(test_probes) > 0:
        divider_x = n_benign - 0.5
        ax.axvline(x=divider_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Section labels
        ax.text(n_benign / 2 - 0.5, 107, 'Train Probes', ha='center', color='white', fontsize=11)
        ax.text(n_benign + len(test_probes) / 2 - 0.5, 107, 'Test Probes', ha='center', color='white', fontsize=11)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, color='white', fontsize=9)
    ax.set_ylabel('TPR @ 1% FPR', color='white', fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(f'Probe Detection Performance ({probe_type.title()} Probes)', color='white', fontsize=14, pad=20)
    ax.legend(loc='lower left', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Results Summary ({probe_type} probes)")
    print(f"{'='*50}")
    for name, b, t in zip(all_names, all_baselines, all_triggered):
        drop = b - t
        print(f"{name.replace(chr(10), ' ')}: {b:.0f}% → {t:.0f}% (drop: {drop:.0f}pp)")


# Usage
if __name__ == "__main__":
    harm_json = "../outputs/harm_eval_v3.json"
    benign_json = "../outputs/benign_eval_v3.json"
    
    for probe_type in ["logistic", "mlp", "attention"]:
        plot_probe_detection(
            harm_json_path=harm_json,
            benign_json_path=benign_json,
            probe_type=probe_type,
            output_path=f"probe_detection_{probe_type}_v3.png"
        )




    # Just test probes
#plot_probe_detection("safety_results.json", probe_type="logistic")

# Both train and test
    #plot_probe_detection(
    #    test_json_path="safety_results.json",
    #    benign_json_path="benign_results.json",
#    probe_type="logistic"
#)

# Compare all probe types
    #for ptype in ["logistic", "mlp", "attention"]:
    # plot_probe_detection(
    #    "safety_results.json",
    #    probe_type=ptype,
    #    output_path=f"results_{ptype}.png"
#)
