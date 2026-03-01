import torch
from models import multimodalIntraInterModal


def _strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        return {k.replace("module.","",1):v for k,v in state_dict.items()}
    return state_dict


def load_model(
        device="cuda",
        model_path:str=None,
        num_classes=6,
        cnn_model_name="densenet169",
        attention_mecanism="gfcam",
        vocab_size=91,
        num_heads=8,
        n=2,
        text_model_name="one-hot-encoder",
        unfreeze_weights="frozen_weights"
    ):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=num_classes, device=device, cnn_model_name=cnn_model_name,
        text_model_name=text_model_name, vocab_size=vocab_size, num_heads=num_heads,
        attention_mecanism=attention_mecanism, n=n, unfreeze_weights=unfreeze_weights
    )

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device=device).eval()
    return model

def find_last_conv(module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv