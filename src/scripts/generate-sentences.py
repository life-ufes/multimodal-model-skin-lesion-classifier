from transformers import pipeline

# Load the T5 model for text generation
text_generator = pipeline("text2text-generation", model="Qwen/Qwen2.5-1.5B")

def convert_to_sentence(age, gender, bleed, hurt, itch, smoke, skin_cancer_history):
    """
    Convert structured data into a detailed clinical medical description.
    """
    prompt = f"""You are a medical assistant. Given the following structured data, generate a detailed and clinical medical description:
    - Age: {age} years
    - Bleeding: {"Yes" if bleed else "No"}
    - Pain: {"Yes" if hurt else "No"}
    - Gender: {gender}
    - Itching: {"Yes" if itch else "No"}
    - Smoker: {"Yes" if smoke else "No"}
    - Skin cancer history: {"Yes" if skin_cancer_history else "No"}

    Consider the patient's symptoms and their lifestyle factors in your description. Generate a short, concise clinical summary of the patient's condition.
    """
    response = text_generator(prompt, max_new_tokens=512, num_return_sequences=1, temperature=0.7)
    print(response)
    return response[0]["generated_text"]

# Example usage
sentence = convert_to_sentence(89, "FEMALE", True, True, True, True, True)
print("Generated Sentence:\n", sentence)
