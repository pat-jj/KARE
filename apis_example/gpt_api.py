import os
from openai import OpenAI

with open("./apis/openai.key", 'r') as f:
    key = f.readlines()[0].strip()


client = OpenAI(api_key=key)


# gpt-4-turbo-2024-04-09
def get_gpt_response(prompt, model="gpt-4o", seed=44):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        # max_tokens=200,
        temperature=0,
        seed=seed
    ).choices[0].message.content
    return response