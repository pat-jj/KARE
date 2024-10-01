from spliter import split_by_patient
import pickle
import json
from tqdm import tqdm
import os
import time
import botocore
from apis.claude_api import get_claude_response
import concurrent.futures
import threading
import random

MAX_RETRIES = 5 
TIMEOUT_SECONDS = 10  
NUM_THREADS = 30

TASKS = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions

2. Procedures:
   - Invasiveness and complexity of recent procedures 
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)

3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.

"""
    },
    "readmission": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days

Key Considerations:  
1. Conditions:
   - Chronic diseases with high risk of exacerbation (e.g., COPD, heart failure)
   - Conditions requiring close monitoring or frequent adjustments (e.g., diabetes)
   - Recent acute conditions with potential for complications

2. Procedures:
   - Recent major surgeries or interventions with high complication rates
   - Procedures that require extensive follow-up care
   - Incomplete or partially successful procedures

3. Medications:  
   - New medication regimens that may require adjustment
   - Medications with narrow therapeutic windows or high risk of side effects
   - Complex medication schedules that may lead to adherence issues

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.

"""
    },
}


TASKS_ABBR = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit.
Labels: 1 = mortality, 0 = survival

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.

"""
    },
    "readmission": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
    
    """
        },
    
}

LABEL_MAPPING = {
    "mortality": {
        0: "0 (Survival in the subsequent visit)",
        1: "1 (Mortality in the subsequent visit)"
    },
    "readmission": {
        0: "0 (No Readmission within 15 days)",
        1: "1 (Readmission within 15 days)"
    },
}



def generate_reasoning(patient_context, task, ground_truth, medical_knowledge, similar_patient=None):
    
    context = patient_context
    # supplementary_info = format_knowledge(medical_knowledge)
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)
    
    prompt = f"""
Given the following task description, patient EHR context, similar patients, retrieved medical knowledge, and ground truth label, provide a step-by-step reasoning process that leads to the correct prediction:

========================================
# Task #
{TASKS[task]['description']}

========================================
# Patient EHR Context #

{context}

========================================
# Similar Patients #

{" ".join(similar_patients)}

========================================
# Retrieved Medical Knowledge #

{medical_knowledge}

========================================
# Ground Truth #

{ground_truth}

========================================

Please provide a step-by-step reasoning process that leads to the correct prediction based on the patient's context, similar patients, and the retrieved relevant medical knowledge.

**The reasoning chain should follow this structured format:**

1. **Patient Overview**: Go over the key information in the patient's EHR context, with the **Key Considerations** from the task description in mind.
2. **Relevant Retrieved Medical Knowledge**: Highlight the retrieved medical knowledge pertinent to the patient's condition.
3. **Comparison with Similar Patients**: Analyze the similarities and differences between the patient and similar patients, explaining how these factors influence the prediction.
4. **Reasoning Towards Prediction**: Integrate the above information to logically reason towards the predicted outcome.
5. **Conclusion**: Summarize the reasoning and state the prediction without mentioning the ground truth label.

The reasoning should be comprehensive, medically sound, and clearly explain how the patient's information leads to the predicted outcome.

**Important Notes:**
- **Do not mention the ground truth label in the reasoning process**.
- Use the relevant knowledge as needed, but **the main focus should be on the patient's EHR context**.
- Analyze the similarities and differences between the patient and similar patients to justify the prediction.

After generating the reasoning chain, please review it and indicate your confidence in the reasoning chain at the end.

Options of confidence: [Very Confident, Confident, Neutral, Not Confident, Very Not Confident.]

**Output Format:**

# Reasoning Chain #

1. **Patient Overview**:
   [YOUR OUTPUT]

2. **Relevant Retrieved Medical Knowledge**:
   [YOUR OUTPUT]

3. **Comparison with Similar Patients**:
   [YOUR OUTPUT]

4. **Reasoning Towards Prediction**:
   [YOUR OUTPUT]

5. **Conclusion**:
   [YOUR OUTPUT]

# Confidence #
[CONFIDENCE (choose one: "Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident")]
"""
    # print(prompt)

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # response = get_claude_response(llm="sonnet", prompt=prompt)
            response = get_claude_response(llm="opus", prompt=prompt)

            return response
        except botocore.exceptions.ReadTimeoutError:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(TIMEOUT_SECONDS)
    return response



def construct_input_output(patient_context, task, reasoning, ground_truth, medical_knowledge, similar_patient=None):
    
    context = patient_context
    
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)
    
    input_ = f"""
Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.
After the reasoning process, provide the prediction label (0/1).

========================================
# Task #
{TASKS_ABBR[task]['description']}

========================================
# Patient EHR Context #

{context}

========================================
# Similar Patients #

{" ".join(similar_patients)}

========================================
# Retrieved Medical Knowledge #

{medical_knowledge}

"""

    output_ = f"""
# Reasoning #
{reasoning}

# Prediction #
{ground_truth}"""


    return input_, output_


def process_patient(patient_id, patient_context, task, ground_truth, patient_id_train, patient_id_val, patient_id_test, medical_knowledge=None, similar_patient=None):
    print(f"Processing patient {patient_id}...")
    
    print(f"Generating reasoning for patient {patient_id}... size of similar_patient: {len(similar_patient)}")
    reasoning = generate_reasoning(patient_context, task, ground_truth, medical_knowledge, similar_patient)
    
    if (patient_id in patient_id_train or patient_id in patient_id_val) and "\n# Confidence #\nNot Confident" in reasoning:
        return None
    reasoning = reasoning.split("\n# Confidence #\n")[0]
    reasoning = reasoning.replace("# Reasoning Chain #\n", "")
    input_, output_ = construct_input_output(patient_context, task, reasoning, ground_truth, medical_knowledge, similar_patient=similar_patient)
    item = {"input": input_, "output": output_}
    

    if patient_id in patient_id_train:
        return ("train", item)
    elif patient_id in patient_id_val:
        return ("val", item)
    elif patient_id in patient_id_test:
        return ("test", item)
    else:
        return None

def process_dataset(dataset, task, i):
    ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}.pkl"
    context_dir = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{dataset}_{task}.json"
    save_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_ulti"
    medical_knowledge_path = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/patient_to_top_10_knowledge_node_hits.json"
    similar_patient_path = f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_1_patient_contexts_{dataset}_{task}.json"
    patient_data_path =  f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{dataset}_{task}.json"
    test_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_test.json"
    train_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_train.json"
    val_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_val.json"
    
    
    patient_contexts = json.load(open(context_dir, "r"))
    medical_knowledge = json.load(open(medical_knowledge_path, "r"))
    similar_patient = json.load(open(similar_patient_path, "r"))
    patient_data = json.load(open(patient_data_path, "r"))
    test_sample = json.load(open(test_sample_path, "r"))
    train_sample = json.load(open(train_sample_path, "r"))
    val_sample = json.load(open(val_sample_path, "r"))
    
    patient_id_test = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in test_sample]
    patient_id_train = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in train_sample]
    patient_id_val = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in val_sample]
    patient_id_all = patient_id_train + patient_id_val + patient_id_test
    random.shuffle(patient_id_all)
    
    
    train_data, val_data, test_data = [], [], []
    
    def save_data(i):
        with open(f"{save_path}/{dataset}_{task}_train_{i}.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
            for item in val_data:
                f.write(json.dumps(item) + "\n")
        
        with open(f"{save_path}/{dataset}_{task}_val_{i}.jsonl", "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")
        
        with open(f"{save_path}/{dataset}_{task}_test_{i}.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
                
    
    lock = threading.Lock()
    
    def process_batch(batch):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            future_to_patient = {executor.submit(process_patient, patient_id, patient_contexts[patient_id], task, patient_data[patient_id]["label"], patient_id_train, patient_id_val, patient_id_test, medical_knowledge, similar_patient[patient_id]): patient_id for patient_id in batch}
            for future in concurrent.futures.as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    print(f'{patient_id} generated an exception: {exc}')
        return results
    
    batches = [patient_id_all[i:i+NUM_THREADS] for i in range(0, len(patient_id_all), NUM_THREADS)]
    # batches = [patient_id_all[i:i+NUM_THREADS] for i in range(0, 200, NUM_THREADS)]
    
    for batch in tqdm(batches, desc="Processing batches"):
        batch_results = process_batch(batch)
        with lock:
            for result in batch_results:
                if result[0] == "train":
                    train_data.append(result[1])
                elif result[0] == "val":
                    val_data.append(result[1])
                elif result[0] == "test":
                    test_data.append(result[1])
        
        save_data(i)
    
    save_data(i)
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    for dataset in [
        "mimic3", 
        "mimic4"
        ]:
        for task in [
            "mortality", 
            "readmission"
            ]:
            train_data_combined, val_data_combined, test_data_combined = [], [], []
            for i in range(1):
                print(f"Processing {dataset}_{task}_{i}...")
                train_data, val_data, test_data = process_dataset(dataset, task, i)
                print(f"Finished processing {dataset}_{task}_{i}...")
                train_data_combined.extend(train_data)
                val_data_combined.extend(val_data)
                test_data_combined.extend(test_data)
                
            with open(f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_ulti/{dataset}_{task}_train.jsonl", "w") as f:
                for item in train_data_combined:
                    f.write(json.dumps(item) + "\n")
                for item in val_data_combined:
                    f.write(json.dumps(item) + "\n")
                    
            with open(f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_ulti/{dataset}_{task}_test.jsonl", "w") as f:
                for item in test_data_combined:
                    f.write(json.dumps(item) + "\n")

            print(f"Finished processing {dataset}_{task}...")
            
            
        
                
                
        
        
    