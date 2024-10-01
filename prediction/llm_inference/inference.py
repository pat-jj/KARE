import requests
import json



def get_llm_response(prompt, max_length=1000, temperature=1.0):
    url = "http://localhost:1212/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": 1.0,
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response key in JSON')
    else:
        return f"Error: {response.status_code}, {response.text}"


def get_llm_json_response(prompt, schema, max_tokens=None, temperature=None):
    port = 1212

    url = f"http://localhost:{port}/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "schema": schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response key in JSON')
    else:
        return f"Error: {response.status_code}, {response.text}"



if __name__ == '__main__':
    prompt = "who are you?"
    print(get_llm_response(prompt))