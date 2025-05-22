# Multimodal Model - Skin Lesion Classifier Framework

![Multimodal Model for Skin Lesion Recognition](./images/multimodal_model_representation.png)

This repository contains a multimodal model for classifying skin lesions. The model integrates an image feature extractor (e.g., **VGG16**, **ResNet18**, **ResNet50**, **DenseNet169**, among others) with clinical data to form a powerful multimodal architecture. Also, on the latest updates, many transformers based on models have been included.

## 1. Preparing the Dataset

1. Download the **PAD-20** dataset provided by UFES.
2. Extract the files and move them to the `data/` directory of this project.

> **Note:** Ensure the folder structure is correct for the training script to properly recognize the files.

* This framework supports ISIC-2019, ISIC-2020, PAD-UFES-20 and PAD-UFES20-Extended datasets.

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

# Configuration

Go to the 'conf' folder and create a .env file with the following variables:

`
NUM_EPOCHS: Quantity of epochs to train the models.
BATCH_SIZE: Batch size.
K_FOLDS: Quantity of folds to be created.
LIST_NUM_HEADS: List with the quantity of 'heads' to be used oon the models train process.
COMMON_DIM: Vector size to be used as the projected vector on the visual and the textual feature vectors.
DATASET_FOLDER_NAME: Name of the dataset folder
DATASET_FOLDER_PATH: Folder path of the dataset
UNFREEZE_WEIGHTS: Flag to indicating the freezing state of the weights
LLM_MODEL_NAME_SEQUENCE_GENERATOR: Name of the LLM generator model when trainning with sentences.
RESULTS_FOLDER_PATH: Folder path to indicate where the results will be saved in the end of the trainning process.
`

You can use the '.env-test' file as base to write your own '.env' file.


## 3. Training the Model

![Illustration of Multimodal Model - Using image and metadata](./images/multimodal_model_representation_representation.png)


1. Choose the image feature extractor (such as **VGG16**, **ResNet18**, **ResNet50**, **DenseNet169**, etc.) in the training script.
2. Choose your metadata information preprocessor. It can be 'one-hot-encoder', "tab-transformer" or "bert-base-uncased". 

3. Run the following command:  
   ```bash
   python3 src/train_pad_20.py

   ```
4. Monitor the training process and metrics via **MLFlow** (if configured in the script).

## 4.1 Training a model using ISIC 2019 dataset

1. Change the dataset folder path diretory on 'preprocess_isic_2019.py' script and then run it to create the "metadata.csv" equivalent to this dataset.

2. Then, with the metadata.csv created, run the 'train_isic_2019.py' script to train the defined model.

## 4.2 Create your own mulmodalmodal model

You can create yor own models. Take the created model scripts as examples on the folder 'src/models'. Then, you import it to your 'src/train_pad_20.py' script.


## 5. Plotting the Model

1. Set the path to the model you want to visualize in the script.
2. Run the plot script:  
   ```bash
   python3 src/plot_model.py
   ```
3. A plot or interactive graph of the model will be generated, depending on your configuration.

## 6. Exporting the Model to ONNX Format

1. In the `src/export_model_onnx.py` script, update the `model_path` variable to the desired model path.
2. Then, execute:  
   ```bash
   python3 src/export_model_onnx.py
   ```
3. This will generate an ONNX file (default: `multimodal_model.onnx`).

> **Warning:** Do not alter the model architecture before exporting it. If changes are necessary, ensure you know the original architecture used for training, as any modifications might prevent proper conversion.

## 7. Visualizing the Converted Model

To inspect the ONNX file structure, use [Netron](https://netron.app/):  
```bash
netron multimodal_model.onnx
```
This will launch a graphical interface to explore the model’s layers.

## 8. Visualizing the MLFlow UI

To track training metrics, model versions, and saved artifacts, run:  
```bash
mlflow ui
```
The interface will be available at [http://localhost:5000](http://localhost:5000) (default port).

---

### Contact

For questions or suggestions, feel free to open an **issue** or submit a **pull request**. Let us know how you’re using or improving this repository!