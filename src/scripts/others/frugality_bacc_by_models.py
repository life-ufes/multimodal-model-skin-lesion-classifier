import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "Feature Extractor": [
        "mvitv2_small", "coat_lite_small", "davit_tiny",
        "caformer_b36", "beitv2_large_patch16_224",
        "nextvit_small", "vgg16", "mobilenet-v2", "densenet169", "resnet-50"
    ],
    "BACC (%)": [
        0.8137, 0.7919, 0.8228, 0.8062, 0.8074, 0.8067, 0.7776, 0.7999, 0.8048, 0.8164
    ],
    "Parameters (M)": [
        26, 22, 6.5, 28, 307, 31.76, 138, 3.4, 14, 25.6
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot with Parameters (M) on the x-axis and BACC on the y-axis
plt.scatter(df["Parameters (M)"], df["BACC (%)"], color='skyblue', s=100, edgecolor='black', label="Models")

# Highlighting the smallest models with greater BACC (e.g., models with <= 10M parameters)
highlight = df[df["Parameters (M)"] <= 10]

# Plotting the highlighted points (small models with great BACC)
plt.scatter(highlight["Parameters (M)"], highlight["BACC (%)"], color='red', s=150, edgecolor='black', label="Small & High BACC Models")

# Adding labels and title
plt.xlabel("Parameters (M)")
plt.ylabel("Balanced Accuracy (%)")
plt.title("Small Models with High BACC")

# Annotate each point with the model's name
for i in range(len(df)):
    plt.annotate(df["Feature Extractor"][i], 
                 (df["Parameters (M)"][i], df["BACC (%)"][i]),
                 textcoords="offset points", 
                 xytext=(5, 5),  # offset the label slightly to avoid overlap
                 ha='center', 
                 fontsize=8)

# Annotate the highlighted small models with greater BACC
for i in range(len(highlight)):
    plt.annotate(highlight["Feature Extractor"].iloc[i], 
                 (highlight["Parameters (M)"].iloc[i], highlight["BACC (%)"].iloc[i]),
                 textcoords="offset points", 
                 xytext=(5, 5),  # offset the label slightly to avoid overlap
                 ha='center', 
                 fontsize=8, 
                 color='black')

# Display the legend
plt.legend()

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
