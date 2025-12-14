import requests
import re
import logging

def request_to_ollama(
    prompt,
    model_name="qwen:0.5b",
    host="http://localhost:11434",
    thinking: bool = False,
    timeout: int = 120
):
    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": thinking
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.Timeout:
        logging.warning("Timeout ao consultar o Ollama.")
        return None
    except Exception as e:
        logging.error(f"Erro ao consultar o modelo: {e}")
        return None

def filter_generated_response(generated_sentence: str) -> str:
    """
    Extrai o PRIMEIRO objeto JSON válido da resposta do LLM.
    Compatível com <think>, texto extra e múltiplos blocos.
    """

    if not generated_sentence:
        logging.info("Resposta vazia do LLM.")
        return None

    # Remove bloco <think> se existir
    if "</think>" in generated_sentence:
        generated_sentence = generated_sentence.split("</think>", 1)[1]

    # Regex para capturar o primeiro JSON {...}
    json_match = re.search(r"\{[\s\S]*\}", generated_sentence)

    if not json_match:
        logging.info(
            "Nenhum objeto JSON encontrado na resposta do LLM."
        )
        return None


    return json_match.group(0).strip()
