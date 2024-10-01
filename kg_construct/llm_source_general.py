import csv
from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response
import json
import re
from tqdm import tqdm
import os

# LLM = "GPT"
LLM = "Claude"

def read_code2name(file_path):
    condition_mapping_file = f"{file_path}/CCSCM.csv"
    procedure_mapping_file = f"{file_path}/CCSPROC.csv"
    drug_file = f"{file_path}/ATC.csv"

    condition_dict = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()

    procedure_dict = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()

    drug_dict = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()
                
    return condition_dict, procedure_dict, drug_dict
            

def extract_data_in_brackets(input_string):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches

def divide_text(long_text, max_len=800):
    sub_texts = []
    start_idx = 0
    while start_idx < len(long_text):
        end_idx = start_idx + max_len
        sub_text = long_text[start_idx:end_idx]
        sub_texts.append(sub_text)
        start_idx = end_idx
    return sub_texts
            

def graph_gen(term: str, mode: str):
    if mode == "condition":
        example = \
"""Example:
prompt: systemic lupus erythematosus
updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for, systemic lupus erythematosus], [systemic lupus erythematosus, is treated with, steroids], [methylprednisolone, is a, steroid]]
"""
    elif mode == "procedure":
        example = \
"""Example:
prompt: endoscopy
updates: [[endoscopy, is a, medical procedure], [endoscopy, used for, diagnosis], [endoscopic biopsy, is a type of, endoscopy], [endoscopic biopsy, can detect, ulcers]]
"""
    elif mode == "drug":
        example = \
"""Example:
prompt: iobenzamic acid
updates: [[iobenzamic acid, is a, drug], [iobenzamic acid, may have, side effects], [side effects, can include, nausea], [iobenzamic acid, used as, X-ray contrast agent], [iobenzamic acid, formula, C16H13I3N2O3]]
"""

    prompt = \
f"""Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.
The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
Both ENTITY 1 and ENTITY 2 should be noun.
Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 200 (maximum).

{example}

prompt: {term}
updates:
"""

    print("Prompting the LLM ...")
    if LLM == "GPT":
        response = get_gpt_response(
            prompt=prompt,
            )
    elif LLM == "Claude":
        response = get_claude_response(
            llm="sonnet",
            prompt=prompt
        )   
    else:
        raise ValueError(f"Unknown LLM: {LLM}")
    
    print(response)
    # json_string = str(response)
    # json_data = json.loads(json_string)

    triples = extract_data_in_brackets(response)
    outstr = ""
    for triple in triples:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'

    return outstr



def main():
    resource_path = "./resources"
    condition_dict, procedure_dict, drug_dict = read_code2name(resource_path)
    
    graph_path = "./graphs"
    for dict_, mode, code in zip([condition_dict, procedure_dict, drug_dict], \
        ["condition", "procedure", "drug"], ["CCSCM", "CCSPROC", "ATC3"]):
        
        print(f"Generating {mode} graphs (code: {code}) ...")
        
        for key in tqdm(dict_.keys()):
            file = f'{graph_path}/{mode}/{LLM}/{key}.txt'
            if os.path.exists(file):
                with open(file=file, mode="r", encoding='utf-8') as f:
                    prev_triples = f.read()
                if len(prev_triples.split('\n')) < 300:
                    outstr = graph_gen(term=dict_[key], mode=mode)
                    outfile = open(file=file, mode='w', encoding='utf-8')
                    outstr = prev_triples + outstr
                    # print(outstr)
                    outfile.write(outstr)
            else:
                outstr = graph_gen(term=dict_[key], mode=mode)
                outfile = open(file=file, mode='w', encoding='utf-8')
                outstr = outstr
                outfile.write(outstr)
        
        print(f"Finished generating {mode} graphs (code: {code})")
        
        
if __name__ == "__main__":
    main()
            
    