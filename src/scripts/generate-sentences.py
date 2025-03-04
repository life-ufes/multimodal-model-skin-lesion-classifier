from transformers import pipeline
import torch
import pandas as pd
import os

from transformers import pipeline
import torch
import pandas as pd
import os

def preprocess_patient_data(row, column_names):
    """
    Processa os dados do paciente e formata como uma string legível,
    considerando todas as colunas disponíveis.
    """
    # Filtra os dados que não são vazios ou nulos
    filtered_data = {col: str(row[col]) for col in column_names if pd.notna(row[col]) and row[col] != ""}
    
    # Mapeamento das colunas para suas descrições formatadas
    column_descriptions = {
        "patient_id": "Patient ID: {}",
        "lesion_id": "Lesion ID: {}",
        "age": "Age: {} years old",
        "gender": "Gender: {}",
        "region": "Lesion located in {}",
        "diameter_1": "Lesion size: {}mm",
        "diameter_2": "Lesion secondary size: {}mm",
        "smoke": "Smoker: {}",
        "drink": "Drinks alcohol: {}",
        "background_father": "Family history (father): {}",
        "background_mother": "Family history (mother): {}",
        "skin_cancer_history": "History of skin cancer: {}",
        "cancer_history": "Other cancer history: {}",
        "has_piped_water": "Has piped water: {}",
        "has_sewage_system": "Has sewage system: {}",
        "fitspatrick": "Fitzpatrick skin type: {}",
        "pesticide": "Exposure to pesticide: {}",
        "itch": "Experiencing itching: {}",
        "grew": "Lesion has grown: {}",
        "hurt": "Experiencing pain: {}",
        "changed": "Lesion has changed: {}",
        "bleed": "Lesion is bleeding: {}",
        "elevation": "Lesion is elevated: {}",
        "img_id": "Image ID: {}"
    }
    
    # Construindo a descrição final apenas para colunas presentes no conjunto de dados
    description_parts = [
        column_descriptions[col].format(filtered_data[col])
        for col in filtered_data if col in column_descriptions
    ]

    return ". ".join(description_parts)

def convert_to_sentence(original_input: str, columns_names: list, model_name: str, device: str = "cpu"):
    """
    Converte os dados estruturados (pré-processados) em uma descrição clínica detalhada.
    """
    try:
        # Configura o device para o pipeline: 0 para GPU ou -1 para CPU
        device="cuda" if torch.cuda.is_available() else "cpu"
        text_generator = pipeline("text-generation", model=model_name, device=device)
        
        # Formata a lista de colunas para referência no prompt
        # prompt = f"Just summarize the patient's information and its skin lesion condition based on the following details. Data: {original_input}"
        prompt=f"Generate a detailed clinical summary of a patient with a skin lesion based on the following structured information: {original_input}. Ensure the summary is concise, medically relevant, and written in a professional tone. Include details about the patient's demographics, medical history, lesion characteristics, and potential risk factors."
        response = text_generator(prompt, max_new_tokens=128)
        generated_sentence = response[0]['generated_text'].replace(prompt, "").strip()
        return generated_sentence
    except Exception as e:
        print(f"Erro ao tentar gerar a sentença. Erro: {e}")
        return None

def load_dataset(file_path: str):
    """
    Carrega os dados de um arquivo CSV e retorna os nomes das colunas e o DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Erro ao ler o arquivo. Erro: {e}")
        return None, None

if __name__ == "__main__":
    # Caminho do arquivo com os dados dos pacientes
    file_folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/PAD-UFES-20"
    
    # Define o device para execução (GPU se disponível, senão CPU)
    device = "cpu"  # ou "cuda" se GPU estiver disponível
    
    # Define o modelo a ser utilizado
    model_name = "Qwen/Qwen2.5-1.5B"
    
    sentences = []
    
    # Carrega os dados do arquivo CSV (neste exemplo, o arquivo 'metadata.csv')
    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))
    if file_content is not None:
        # Itera sobre cada paciente (linha do DataFrame)
        for _, row in file_content.iterrows():
            # Pré-processa os dados do paciente para criar uma string descritiva
            preprocessed_data = preprocess_patient_data(row, columns_names)
            print(f"Preprocessing: {preprocessed_data}\n")
            # Utiliza os dados pré-processados para gerar uma sentença com o modelo
            sentence = convert_to_sentence(preprocessed_data, columns_names, model_name, device)
            print(f"Generated Sentence: {sentence}\n")
            sentences.append(f"{sentence}\n")
    
        # Exibe a sentença gerada
        print("Sentenças geradas para cada paciente:")
        for s in sentences:
            print(s)
