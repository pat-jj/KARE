import csv
from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response
import re
from tqdm import tqdm
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed


LLM = "Claude"


def read_code2name(file_path):
    condition_mapping_file = f"{file_path}/CCSCM.csv"
    procedure_mapping_file = f"{file_path}/CCSPROC.csv"
    drug_file = f"{file_path}/ATC.csv"

    condition_dict = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code'].replace(".", "")] = row['name']
            
    condition_dict['19'] = "Lung Cancer"

    procedure_dict = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code'].replace(".", "")] = row['name']

    drug_dict = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name']
                
    return condition_dict, procedure_dict, drug_dict


def extract_data_in_brackets(input_string):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches


def graph_gen(term: str, mode: str, save_path: str):
    if mode == "condition":
        example = \
"""Example:
concept: non-small cell lung cancer
updates: [[non-small cell lung cancer, is a type of, lung cancer], [non-small cell lung cancer, may cause, shortness of breath], [PD-L1 expression, is a biomarker for, non-small cell lung cancer], [non-small cell lung cancer, is treated with, targeted therapy], [EGFR mutation, affects, treatment response in non-small cell lung cancer], [stage of non-small cell lung cancer, influences, mortality risk], [smoking history, increases risk of, non-small cell lung cancer]]
"""
    elif mode == "procedure":
        example = \
"""Example:
concept: low-dose CT screening
updates: [[low-dose CT screening, is used for, early lung cancer detection], [low-dose CT screening, can detect, lung nodules], [annual low-dose CT screening, reduces, lung cancer mortality], [low-dose CT screening, is recommended for, high-risk smokers], [low-dose CT screening, may lead to, false positives], [pulmonary nodule detection, is a primary outcome of, low-dose CT screening], [low-dose CT screening, helps in, lung cancer staging]]
"""
    elif mode == "drug":
        example = \
"""Example:
concept: osimertinib
updates: [[osimertinib, is a, targeted therapy drug], [osimertinib, treats, EGFR-mutated non-small cell lung cancer], [osimertinib, may cause, side effects], [side effects, can include, skin rash], [osimertinib, improves, progression-free survival], [osimertinib, targets, T790M mutation], [resistance to osimertinib, can develop over time], [osimertinib, is used in, advanced stage lung cancer]]
"""

    prompt = \
f"""Given a medical concept (a medical condition, procedure, or drug), extrapolate as many relationships as possible that are relevant to (1) mortality prediction for lung cancer patients, and (2) lung cancer prediction. Provide a list of updates.
The relationships should be helpful for healthcare prediction, focusing on lung cancer mortality and risk factors.
Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
Both ENTITY 1 and ENTITY 2 should be noun phrases.
Do this in both breadth and depth, considering factors that influence lung cancer development, progression, and patient outcomes.
Expand the list of [ENTITY 1, RELATIONSHIP, ENTITY 2] as much as you can (ideal size: 200).

{example}

concept: {term}
updates:
"""

    print("Prompting the LLM ...")
    if LLM == "GPT":
        response = get_gpt_response(
            prompt=prompt,
            )
    elif LLM == "Claude":
        response = get_claude_response(
            llm="opus",
            prompt=prompt
        )   
    else:
        raise ValueError(f"Unknown LLM: {LLM}")
    
    # json_string = str(response)
    # json_data = json.loads(json_string)

    triples = extract_data_in_brackets(response)
    outstr = ""
    for triple in triples:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'
        
    # Write the output to file within this function
    with open(save_path, "a") as f:
        f.write(outstr)

    return outstr



def process_item(item, item_dict, mode, item_set):
    save_path = f'./theme_specific_graphs/{mode}/{LLM}/{item}.txt'
    if item not in item_set:
        read_path = f"./graphs/{mode}/{LLM}/{item}.txt"
        with open(read_path, "r") as f:
            lines = f.readlines()[:50]
        with open(save_path, "w") as f:
            for line in lines:
                f.write(line)
    else:
        graph_gen(term=item_dict[item], mode=mode, save_path=save_path)

def main():
    condition_dict, procedure_dict, drug_dict = read_code2name("./resources")
    
    lung_mort_dataset_path = "/shared/eng/pj20/kelpie_exp_data/ehr_data/mimic3_lung_mortality.pkl"
    with open(lung_mort_dataset_path, "rb") as f:
        lung_mort_dataset = pickle.load(f)
        
    samples = [sample for sample in lung_mort_dataset.samples if sample['label'] != 1000]
    lung_mort_dataset.samples = samples
    
    condition_set = lung_mort_dataset.get_all_tokens('conditions')
    procedure_set = lung_mort_dataset.get_all_tokens('procedures')
    drug_set = lung_mort_dataset.get_all_tokens('drugs')
    
    # Create directories if they don't exist
    for mode in ['condition', 'procedure', 'drug']:
        os.makedirs(f'./theme_specific_graphs/{mode}/{LLM}', exist_ok=True)

    # Process all items concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        print("Generating theme-specific (lung cancer) graphs...")
        for condition in condition_dict.keys():
            futures.append(executor.submit(process_item, condition, condition_dict, 'condition', condition_set))
        
        for procedure in procedure_dict.keys():
            futures.append(executor.submit(process_item, procedure, procedure_dict, 'procedure', procedure_set))
        
        for drug in drug_dict.keys():
            futures.append(executor.submit(process_item, drug, drug_dict, 'drug', drug_set))

        # Show progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Done!")

if __name__ == "__main__":
    main()
        