```markdown
# Multimodal Model - Skin Lesion Classifier

This repository contains a multimodal model for classifying skin lesions. The model integrates an image feature extractor (e.g., **VGG16**, **ResNet18**, **ResNet50**, **DenseNet169**, among others) with clinical data to form a powerful multimodal architecture.

## 1. Preparing the Dataset

1. Download the **PAD-20** dataset provided by UFES.
2. Extract the files and move them to the `data/` directory of this project.

> **Note:** Ensure the folder structure is correct for the training script to properly recognize the files.

## 2. Setting Up the Environment

1. Create a new Conda environment:  
   ```bash
   conda create -n multimodal-env
   ```
2. Activate the newly created environment:  
   ```bash
   conda activate multimodal-env
   ```
3. Install the required dependencies (if not done already).  
   > **Tip:** If a `requirements.txt` file is available, simply run:  
   > ```bash
   > pip install -r requirements.txt
   > ```
   Otherwise, install the libraries manually as mentioned in the documentation.

## 3. Training the Model

1. Choose the image feature extractor (such as **VGG16**, **ResNet18**, **ResNet50**, **DenseNet169**, etc.) in the training script.
2. Run the following command:  
   ```bash
   python3 src/train.py
   ```
3. Monitor the training process and metrics via **MLFlow** (if configured in the script).

## 4. Plotting the Model

1. Set the path to the model you want to visualize in the script.
2. Run the plot script:  
   ```bash
   python3 src/plot_model.py
   ```
3. A plot or interactive graph of the model will be generated, depending on your configuration.

## 5. Exporting the Model to ONNX Format

1. In the `src/export_model_onnx.py` script, update the `model_path` variable to the desired model path.
2. Then, execute:  
   ```bash
   python3 src/export_model_onnx.py
   ```
3. This will generate an ONNX file (default: `multimodal_model.onnx`).

> **Warning:** Do not alter the model architecture before exporting it. If changes are necessary, ensure you know the original architecture used for training, as any modifications might prevent proper conversion.

## 6. Visualizing the Converted Model

To inspect the ONNX file structure, use [Netron](https://netron.app/):  
```bash
netron multimodal_model.onnx
```
This will launch a graphical interface to explore the model’s layers.

## 7. Visualizing the MLFlow UI

To track training metrics, model versions, and saved artifacts, run:  
```bash
mlflow ui
```
The interface will be available at [http://localhost:5000](http://localhost:5000) (default port).

---

### Contact

For questions or suggestions, feel free to open an **issue** or submit a **pull request**. Let us know how you’re using or improving this repository!