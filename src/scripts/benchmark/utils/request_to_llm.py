import requests
import json
import re

def request_to_ollama(
    prompt, model_name="qwen3:0.6b",
    host="http://localhost:11434", thinking: bool = False):
    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": thinking,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        print(f"Erro ao consultar o modelo: {e}")
        return ""

def filter_generated_response(generated_sentence: str) -> str:
    """
    Extrai o PRIMEIRO objeto JSON válido da resposta do LLM.
    Compatível com <think>, texto extra e múltiplos blocos.
    """

    if not generated_sentence:
        raise ValueError("Resposta vazia do LLM.")

    # Remove bloco <think> se existir
    if "</think>" in generated_sentence:
        generated_sentence = generated_sentence.split("</think>", 1)[1]

    # Regex para capturar o primeiro JSON {...}
    json_match = re.search(r"\{[\s\S]*\}", generated_sentence)

    if not json_match:
        raise ValueError(
            "Nenhum objeto JSON encontrado na resposta do LLM."
        )

    return json_match.group(0).strip()
