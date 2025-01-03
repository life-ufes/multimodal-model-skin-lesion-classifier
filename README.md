# multimodal-model-skin-lesion-classifier

Baixe o dataset da ufes (PAD-20) e após extraí-las adicioná-las dentro da pasta 'data'.
To train the models:
-- Choose the image feature extractor such as VGG16, ResnEt18, Resnet50, DenseNet169 or other.
-- Choose your text encoder.
Then run:
`python3 src/train.py`

To plot the model:
Choose the model path and then run the command bellow:
`python3 src/plot_model.py`

Export model on onnx format:
Chaange the model_path to the wanted one then run the command:
`python3 src/export_model_onnx.py `

To visualize the converted model run:
`netron multimodal_model.onnx`