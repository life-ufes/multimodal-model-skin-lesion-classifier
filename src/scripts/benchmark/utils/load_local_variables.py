import os
from dotenv import load_dotenv

def get_env_variables():
    # Caminho absoluto para o arquivo .env na pasta /config
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../conf/.env'))
    
    # Carrega as variáveis de ambiente
    load_dotenv(dotenv_path)

    # Retorna um dicionário com as variáveis relevantes
    return {
        "num_epochs": int(os.getenv("NUM_EPOCHS", 10)),
        "batch_size": int(os.getenv("BATCH_SIZE", 32)),
        "k_folds": int(os.getenv("K_FOLDS", 5)),
        "common_dim": int(os.getenv("COMMON_DIM", 128)),
        "list_num_heads": eval(os.getenv("LIST_NUM_HEADS", "[2, 4, 8]")),
        "dataset_folder_name": os.getenv("DATASET_FOLDER_NAME", "PAD-UFES-20"),
        "dataset_folder_path": os.getenv("DATASET_FOLDER_PATH"),
        "results_folder_path": os.getenv("RESULTS_FOLDER_PATH"),
        "unfreeze_weights": os.getenv("UNFREEZE_WEIGHTS", "False").lower() in ("true", "1", "yes"),
        "llm_model_name_sequence_generator": os.getenv("LLM_MODEL_NAME_SEQUENCE_GENERATOR", None)
    }