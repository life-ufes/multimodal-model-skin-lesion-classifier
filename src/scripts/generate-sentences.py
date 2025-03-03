from transformers import pipeline
import torch

def convert_to_sentence(original_input, text_generator):
    try:
        """
        Convert structured data into a detailed clinical medical description.
        """
        age, gender, bleed, hurt, itch, smoke, skin_cancer_history, cancer_history = original_input
        prompt = f"Summarize this patient's skin lesion condition in one sentence: Age {age}, {gender}, Bleeding: {bleed}, Pain: {hurt}, Itching: {itch}, Smoker: {smoke}, Skin cancer history: {skin_cancer_history}, Other cancer history: {cancer_history}."

        response = text_generator(prompt, max_new_tokens=512, truncation=True, temperature=0.7)
        generated_sentence = response[0]['generated_text'].replace(prompt, "").strip()
        return generated_sentence
    except Exception as e:
        print(f"Erro ao tentar gerar um parágrafo. Erro:{e}\n")
    finally:
        pass

if __name__=="__main__":
    # Usar o device detectado, caso tenha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lista de modelos que podem ser usados
    list_of_models = ["facebook/bart-large-cnn", "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B"]
    orinal_input=(89, "FEMALE", True, True, True, True, True, True)
    sentences=[]
    for model_name in list_of_models:
        text_generator = pipeline("text2text-generation", model=model_name, device=device)
        sentence = convert_to_sentence(orinal_input, text_generator)
        sentences.append(f"Model used: {model_name}. Generated Sentence: {sentence}\n")
    
    # Printar as sentenças de cada modelo usado
    print(sentences)