import numpy as np
from scipy.stats import wilcoxon

# Define the metrics (mean and std dev) for each model
metrics = {
    "Our Model": {"ACC": (0.7916, 0.0245), "BACC": (0.7826, 0.0316), "AUC": (0.9295, 0.0163)},
    "No Metadata": {"ACC": (0.616, 0.051), "BACC": (0.651, 0.050), "AUC": (0.901, 0.007)},
    "Concatenation": {"ACC": (0.741, 0.014), "BACC": (0.728, 0.029), "AUC": (0.929, 0.006)},
    "MetaBlock": {"ACC": (0.735, 0.013), "BACC": (0.765, 0.017), "AUC": (0.935, 0.004)},
    "MetaNet": {"ACC": (0.732, 0.054), "BACC": (0.742, 0.019), "AUC": (0.936, 0.006)},
    "MD-Net": {"ACC": (0.796, 0.0), "BACC": (0.814, 0.0), "AUC": (0.956, 0.0)},
    "Fully-CrossAttention": {"ACC": (0.768, 0.022), "BACC": (0.775, 0.022), "AUC": (0.947, 0.007)},
}

# Generate random samples and perform Wilcoxon test
for model, values in metrics.items():
    if model == "Our Model":
        continue
    print(f"Comparing 'Our Model' with {model}:")
    for metric in ["ACC", "BACC", "AUC"]:
        mean_our, std_our = metrics["Our Model"][metric]
        mean_model, std_model = values[metric]
        
        # Generate samples for both models
        samples_our = np.random.normal(mean_our, std_our, 100)
        samples_model = np.random.normal(mean_model, std_model, 100)
        
        # Wilcoxon test
        stat, p_value = wilcoxon(samples_our, samples_model)
        print(f"{metric}: Wilcoxon stat={stat}, p-value={p_value}")
    print("\n")
