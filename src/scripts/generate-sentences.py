from transformers import pipeline
import torch
import pandas as pd
import os

def preprocess_patient_data(row, column_names):
    """
    Processa os dados do paciente e retorna um dicionário com os valores filtrados.
    """
    filtered_data = {col: str(row[col]) for col in column_names if pd.notna(row[col]) and row[col] != ""}
    return filtered_data

def convert_to_sentence(data, model_name, device="cpu"):
    """
    Converte os dados estruturados do paciente em uma descrição médica utilizando um modelo de geração de texto.
    """
    try:
        text_generator = pipeline("text-generation", model=model_name, device=0 if device == "cuda" else -1)
        
        # Monta o prompt utilizando os valores do dicionário
        prompt = f"""
Please produce a clinical summary in the exact following format:

- Patient ID: {data.get('patient_id', 'N/A')}
- Age: {data.get('age', 'N/A')} years old
- Gender: {data.get('gender', 'N/A')}
- Lesion Location: {data.get('region', 'N/A')}
- Lesion Size: {data.get('diameter_1', 'N/A')} x {data.get('diameter_2', 'N/A')} mm
- Family Medical History:
    - Father: {data.get('background_father', 'N/A')}
    - Mother: {data.get('background_mother', 'N/A')}
- Environmental Factors:
    - Has Piped Water: {data.get('has_piped_water', 'N/A')}
    - Has Sewage System: {data.get('has_sewage_system', 'N/A')}
    - Pesticide Exposure: {data.get('pesticide', 'N/A')}
- Medical History:
    - Skin Cancer History: {data.get('skin_cancer_history', 'N/A')}
    - Family Cancer History: {data.get('cancer_history', 'N/A')}
- Lifestyle:
    - Smoker: {data.get('smoke', 'N/A')}
    - Alcohol Consumption: {data.get('drink', 'N/A')}
- Symptoms:
    - Itching: {data.get('itch', 'N/A')}
    - Growth: {data.get('grew', 'N/A')}
    - Pain: {data.get('hurt', 'N/A')}
    - Changes in Lesion: {data.get('changed', 'N/A')}
    - Bleeding: {data.get('bleed', 'N/A')}
    - Elevation: {data.get('elevation', 'N/A')}

Do not include any extra commentary.
        """

        response = text_generator(prompt, max_new_tokens=256, temperature=0.4, top_p=0.9, repetition_penalty=1.2)
        generated_sentence = response[0]['generated_text'].replace(prompt, "").strip()
        return generated_sentence
    except Exception as e:
        print(f"Erro ao gerar o texto: {e}")
        return None

# Uso:
if __name__ == "__main__":
    file_folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/PAD-UFES-20"
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_dataset(file_path: str):
        try:
            df = pd.read_csv(file_path)
            return df.columns.tolist(), df
        except Exception as e:
            print(f"Erro ao ler o arquivo. Erro: {e}")
            return None, None

    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))
    if file_content is not None:
        for _, row in file_content.iterrows():
            data = preprocess_patient_data(row, columns_names)
            print("="*180)
            print("Dados processados:", data)
            sentence = convert_to_sentence(data, model_name, device)
            print("Generated Sentence:", sentence)
