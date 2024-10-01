import csv
import pickle
from collections import defaultdict
import json
import os
from apis.claude_api import get_claude_response
from apis.gpt_api import get_gpt_response
from tqdm import tqdm
import concurrent.futures
import threading
import time
import botocore

LLM = "Claude"  # or "Claude"
MAX_RETRIES = 5 
TIMEOUT_SECONDS = 10  

def load_mappings():
    condition_mapping_file = "./resources/CCSCM.csv"
    procedure_mapping_file = "./resources/CCSPROC.csv"
    drug_file = "./resources/ATC.csv"

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

def expand_and_map(l, dict_):
    if type(l[0]) == list:
        return [dict_[item] for sublist in l for item in sublist]
    if type(l[0]) == str:
        return [dict_[item] for item in l]

def process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict):
    patient_data = defaultdict(dict)
    patient_data_no_label = defaultdict(dict)
    patient_to_index = sample_dataset.patient_to_index
    
    for patient, idxs in patient_to_index.items():
        for i in range(len(idxs)):
            label = sample_dataset.samples[idxs[i]]['label']
            patient_id = patient + f"_{i}"
            patient_data[patient_id]['label'] = label
            for j in range(i+1):
                idx = idxs[j]
                data = sample_dataset.samples[idx]
                conditions = expand_and_map(data['conditions'], condition_dict)
                procedures = expand_and_map(data['procedures'], procedure_dict)
                drugs = expand_and_map(data['drugs'], drug_dict)
                patient_data[patient_id][f'visit {j}'] = {
                    'conditions': conditions,
                    'procedures': procedures,
                    'drugs': drugs
                }
                patient_data_no_label[patient_id][f'visit {j}'] = {
                    'conditions': conditions,
                    'procedures': procedures,
                    'drugs': drugs   
                }
            
    return patient_data, patient_data_no_label


def transform_patient_data(patient_data):
    output = ""
    for patient_id, visits in patient_data.items():
        output += f"Patient ID: {patient_id}\n\n"
        
        conditions_set = set()
        procedures_set = set()
        medications_set = set()

        for visit_num, visit_data in visits.items():
            if visit_num == "label":
                continue
            output += f"{visit_num.capitalize()}:\n"
            
            output += "Conditions:\n"
            condition_cnt = 1
            for condition in visit_data["conditions"]:
                if condition in conditions_set:
                    output += f"{condition_cnt}. {condition} (continued from previous visit)\n".capitalize()
                elif visit_num != "visit 0":
                    output += f"{condition_cnt}. {condition} (new)\n".capitalize()
                    conditions_set.add(condition)
                else:
                    output += f"{condition_cnt}. {condition}\n".capitalize()
                    conditions_set.add(condition)
                condition_cnt += 1
            
            output += "\nProcedures:\n"
            procedure_cnt = 1
            for procedure in visit_data["procedures"]:
                if procedure in procedures_set:
                    output += f"{procedure_cnt}. {procedure} (continued from previous visit)\n".capitalize()
                elif visit_num != "visit 0":
                    output += f"{procedure_cnt}. {procedure} (new)\n".capitalize()
                    procedures_set.add(procedure)
                else:
                    output += f"{procedure_cnt}. {procedure}\n".capitalize()
                    procedures_set.add(procedure)         
                procedure_cnt += 1
                
            
            output += "\nMedications:\n"
            drug_cnt = 1
            for medication in visit_data["drugs"]:
                if medication in medications_set:
                    output += f"{drug_cnt}. {medication} (continued from previous visit)\n".capitalize()
                elif visit_num != "visit 0":
                    output += f"{drug_cnt}. {medication} (new)\n".capitalize()
                    medications_set.add(medication)
                else:
                    output += f"{drug_cnt}. {medication}\n".capitalize()
                    medications_set.add(medication)
                drug_cnt += 1

            output += "\n"
    
    return output.strip()


# def generate_patient_context(patient_data):
#     prompt = """
# Generate a comprehensive patient context based on the following EHR data. The context should include all crucial information in a unified and consistent format, emphasizing the chronological order of visits. Focus on the patient's medical history, conditions, procedures, and medications across their hospital visits.

# Example input:
# {
#     "10088": {
#         "visit 0": {
#             "conditions": ["septicemia", "shock", "urinary tract infections"],
#             "procedures": ["enteral and parenteral nutrition", "blood transfusion"],
#             "drugs": ["beta blocking agents", "antithrombotic agents"],
#         },
#         "visit 1": {
#             "conditions": ["septicemia", "acute myocardial infarction", "respiratory failure"],
#             "procedures": ["respiratory intubation", "mechanical ventilation"],
#             "drugs": ["antithrombotic agents", "beta blocking agents"],
#         }
#     }
# }

# Example output:
# Patient ID: 10088

# Visit 0:
# Conditions:
# 1. Septicemia
# 2. Shock
# 3. Urinary tract infections

# Procedures:
# 1. Enteral and parenteral nutrition
# 2. Blood transfusion

# Medications:
# 1. Beta blocking agents
# 2. Antithrombotic agents

# Visit 1:
# Conditions:
# 1. Septicemia (continued from previous visit)
# 2. Acute myocardial infarction (new)
# 3. Respiratory failure (new)

# Procedures:
# 1. Respiratory intubation
# 2. Mechanical ventilation

# Medications:
# 1. Antithrombotic agents (continued from previous visit)
# 2. Beta blocking agents (continued from previous visit)

# Summary:
# - Septicemia persisted across both visits.
# - Patient's condition worsened in the second visit, developing acute myocardial infarction and respiratory failure.
# - More intensive procedures (respiratory intubation and mechanical ventilation) were required in the second visit.
# - Core medications (beta blocking agents and antithrombotic agents) were maintained across both visits.

# ==================================================
# Now, generate a similar patient context for the following EHR data, ensuring to maintain the chronological order of visits and highlighting any changes or progressions in the patient's condition:

# Input:
# """ \
# + json.dumps(patient_data, indent=4) + \
# """

# Output:

# """
    
#     # Use the provided function to get the LLM response
#     # response = get_gpt_response(prompt, model="gpt-4o-2024-08-06", max_tokens=4096)
#     retries = 0
#     while retries < MAX_RETRIES:
#         try:
#             if LLM == "GPT":
#                 response = get_gpt_response(prompt=prompt)
#             elif LLM == "Claude":
#                 response = get_claude_response(llm="sonnet", prompt=prompt)
#             else:
#                 raise ValueError(f"Unknown LLM: {LLM}")

#             return response
#         except botocore.exceptions.ReadTimeoutError:
#             retries += 1
#             print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
#             time.sleep(TIMEOUT_SECONDS)

#     # If all retries failed, switch to the other LLM
#     print("Switching to the other LLM...")
#     if LLM == "GPT":
#         response = get_claude_response(llm="sonnet", prompt=prompt)
#     elif LLM == "Claude":
#         response = get_gpt_response(prompt=prompt)
#     else:
#         raise ValueError(f"Unknown LLM: {LLM}")
#     return response
    

def generate_patient_context_threaded(args):
    patient_id, visits = args
    context = transform_patient_data({patient_id: visits})
    return patient_id, context

def base_context_gen(patient_data, out_path, num_threads=30):
    patient_contexts = {}
    count = 0
    print("Generating patient contexts...")
    
    # Create a lock for thread-safe writing to the file
    file_lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_patient = {executor.submit(generate_patient_context_threaded, (patient_id, visits)): patient_id 
                             for patient_id, visits in patient_data.items()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patient_data)):
            patient_id = future_to_patient[future]
            try:
                patient_id, patient_context = future.result()
                patient_contexts[patient_id] = patient_context
                count += 1
                
                # Save every 200 patients
                # if count % 200 == 0:
                #     print(f"Saving contexts for {count} patients...")
                #     with file_lock:
                #         with open(out_path, "w") as f:
                #             json.dump(patient_contexts, f, indent=4)
            except Exception as exc:
                print(f"Patient {patient_id} generated an exception: {exc}")
    
    # Final save
    with open(out_path, "w") as f:
        json.dump(patient_contexts, f, indent=4)
    
    return patient_contexts
            

def main():
    condition_dict, procedure_dict, drug_dict = load_mappings()
    datasets = [
        # "mimic3",
                "mimic4"
                ]
    tasks = [
        "mortality_", 
         "readmission_", 
         "mortality", 
         "readmission", 
         
        #  "lenofstay", 
        #  "drugrec"
             
             ]
    ehr_dir = "/shared/eng/pj20/kelpie_exp_data/ehr_data"
    
    for dataset in datasets:
        for task in tasks:
            print(f"Processing dataset: {dataset}, task: {task}")
            if os.path.exists(f"{ehr_dir}/pateint_{dataset}_{task}.json"):
                print(f"Patient data already exists for dataset: {dataset}, task: {task}, loading from file")
                patient_data = json.load(open(f"{ehr_dir}/pateint_{dataset}_{task}.json", "r"))
                patient_data_no_label = patient_data
            else:
                sample_dataset_path = f"{ehr_dir}/{dataset}_{task}.pkl"
                sample_dataset = pickle.load(open(sample_dataset_path, "rb"))
                patient_data, patient_data_no_label = process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict)
                with open(f"{ehr_dir}/pateint_{dataset}_{task}.json", "w") as f:
                    json.dump(patient_data, f, indent=4)
                    
            # Load or generate patient contexts
            out_path = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{dataset}_{task}.json"
            if os.path.exists(out_path):
                print(f"Patient contexts file exists for dataset: {dataset}, task: {task}, checking completeness")
                with open(out_path, "r") as f:
                    patient_contexts = json.load(f)
                
                # Check if all patients have contexts
                missing_patients = set(patient_data.keys()) - set(patient_contexts.keys())
                if missing_patients:
                    print(f"Found {len(missing_patients)} patients without contexts. Resuming context generation.")
                    new_patient_data = {pid: patient_data[pid] for pid in missing_patients}
                    new_contexts = base_context_gen(new_patient_data, out_path)
                    
                    # Merge new contexts with existing ones
                    patient_contexts.update(new_contexts)
                    
                    # Save updated contexts
                    with open(out_path, "w") as f:
                        json.dump(patient_contexts, f, indent=4)
                    print(f"Updated patient contexts saved for dataset: {dataset}, task: {task}")
                else:
                    print(f"All patients have contexts for dataset: {dataset}, task: {task}")
            else:
                print(f"Generating patient contexts for dataset: {dataset}, task: {task}")
                patient_contexts = base_context_gen(patient_data_no_label, out_path)
                print(f"Patient contexts generated and saved for dataset: {dataset}, task: {task}")
                    
    print("Done!")
    
if __name__ == "__main__":
    main()