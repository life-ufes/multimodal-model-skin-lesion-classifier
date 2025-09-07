import requests
import json

def request_to_ollama(prompt, model_name="qwen:0.5b", host="http://localhost:11434"):
    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Erro ao consultar o modelo: {e}")
        return "{}"  # retorna JSON vazio para não quebrar

def filter_generated_response(generated_sentence:str=None):
    '''
        Filtra as sentenças geradas por LLM
    '''
    try:
        if "</think>" in generated_sentence:
            after_think = generated_sentence.split("</think>", 1)[1].strip()
            print("✅ Extracted text after </think>:\n")
            # Retorno da sentença filtrada
            return after_think
        else:
            print("❌ No </think> found in text.")
            after_think = generated_sentence
        # Retorno da sentença filtrada
        return after_think

    except Exception as e:
        raise ValueError(f"Erro ao realizar a chamada dos dados:{e}\n")