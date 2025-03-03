from transformers import pipeline

# Load the T5 model for text generation
text_generator = pipeline("text2text-generation", model="google/flan-t5-small")

def convert_to_sentence(age, bleed, hurt, itch, smoke):
    """
    Convert structured data into a detailed clinical medical description.
    """
    prompt = f"""You are a medical assistant. Given the following structured data, generate a detailed and clinical medical description:
    - Age: {age} years
    - Bleeding: {"Yes" if bleed else "No"}
    - Pain: {"Yes" if hurt else "No"}
    - Itching: {"Yes" if itch else "No"}
    - Smoker: {"Yes" if smoke else "No"}

    Consider the patient's symptoms and their lifestyle factors in your description. Generate a short, concise clinical summary of the patient's condition.
    """
    response = text_generator(prompt, max_length=512, num_return_sequences=1)
    return response[0]["generated_text"]

# Example usage
sentence = convert_to_sentence(69, True, True, True, False)
print("Generated Sentence:", sentence)
