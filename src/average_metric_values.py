import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dataset_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/stratifiedkfold/2/weighted-after-crossattention/model_densenet169_with_one-hot-encoder_512_with_best_architecture"
    dataset_path = os.path.join(dataset_folder_path, "model_metrics.csv")
    
    # Read the CSV file
    dataset = pd.read_csv(dataset_path, sep=",")
    
    # Define the columns you want to analyze (excluding non-numeric columns)
    numeric_columns = [
        "accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "auc",
        "train_loss", "val_loss", "train process time", "epochs"
    ]
    
    # Ensure that only existing columns are selected
    numeric_columns = [col for col in numeric_columns if col in dataset.columns]
    
    # Convert the selected columns to numeric, coercing errors to NaN
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the mean and standard deviation for the numeric columns
    mean_values = dataset[numeric_columns].mean()
    std_values = dataset[numeric_columns].std()
    
    # Format the result as "avg ± stv" for each metric
    formatted_results = [f"{mean_values[col]:.4f} ± {std_values[col]:.4f}" for col in numeric_columns]
    
    # Create a DataFrame with a single row and metrics as columns
    results_df = pd.DataFrame([formatted_results], columns=numeric_columns)
    
    # Print the new DataFrame
    print("\nFormatted Results DataFrame:")
    print(results_df)
    
    # Optionally, save the results to a CSV file
    results_df.to_csv(f"{dataset_folder_path}/formatted_results.csv", index=False)
