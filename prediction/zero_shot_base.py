import json
import pickle
from spliter import split_by_patient
from apis.claude_api import get_claude_response
from tqdm import tqdm
from collections import defaultdict
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import time
import numpy as np
import concurrent.futures

DATASET = "mimic4"
# TASK = "mortality"
TASK = "readmission"
MODE = "base"
# MODE = "augmented" 
MODEL = "sonnet"

# Constants
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"
LLM = "GPT"  # or "Claude"
MAX_ABSTRACTS = 40
ABSTRACTS_PER_REQUEST = 20  # PubMed recommends no more than 20 IDs per request
ABSTRACTS_PER_CHUNK = 5

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


def prompt_w_base_context(context):
    context = context.split("Summary:")[0]
    input_ = f"""
Given the following task description and patient context, please make a prediction with reasoning based on the patient's context.

# Task # 
{TASKS[TASK]["description"]}
========================================

# Patient Context #
{context}  
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #  
[Your prediction here (1/0)]

Output:
"""

    response = get_claude_response(llm=MODEL, prompt=input_)
    
    return response
    

def prompt_w_augmented_context(patient_context):
    context = patient_context.split("\n\nSupplementary Information:")[0].split("Summary:")[0]
    supplementary_info = patient_context.split("\n\nSupplementary Information:")[1:][0]
    
    input_ = f"""
Given the following task description, patient context, and relevant supplementary information, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

# Task # 
{TASKS[TASK]["description"]}
========================================

# Patient Context #
{context}
========================================

# Supplementary Information #
{supplementary_info}  
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
"""

    response = get_claude_response(llm=MODEL, prompt=input_)
    
    return response


def extract_keywords(text):
    # Placeholder implementation, you can customize this based on your requirements  
    keywords = []
    for word in text.split():
        if len(word) > 6 and word.isalpha():
            keywords.append(word)
    return keywords

def search_pubmed(context: str, retmax: int = 100):
    """Extract keywords from the patient context and search PubMed for related articles."""
    keywords = extract_keywords(context)
    query = " OR ".join(keywords)
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "xml", 
        "sort": "relevance"
    }
    url = PUBMED_SEARCH_URL + "?" + urlencode(params)
    with urlopen(url) as response:
        tree = ET.parse(response)
        root = tree.getroot()
        id_list = root.find("IdList")
        return [id_elem.text for id_elem in id_list.findall("Id")]

def fetch_pubmed_abstracts(pmids):
    """Fetch abstracts for given PubMed IDs."""
    abstracts = []
    for i in range(0, len(pmids), ABSTRACTS_PER_REQUEST):
        batch = pmids[i:i+ABSTRACTS_PER_REQUEST]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract" 
        }
        url = PUBMED_FETCH_URL + "?" + urlencode(params)
        with urlopen(url) as response:
            tree = ET.parse(response)
            root = tree.getroot()
            for article in root.findall(".//Article"):
                abstract = article.find(".//Abstract/AbstractText")
                if abstract is not None and abstract.text:
                    abstracts.append(abstract.text)
        time.sleep(1)  # Respect PubMed's rate limit
    return abstracts

def prompt_w_rag_context(base_context, augmented_context):
    # Search PubMed for a larger set of abstracts using the base context
    pmids = search_pubmed(base_context, retmax=10)
    abstracts = fetch_pubmed_abstracts(pmids)

    # Split the augmented context into context and supplementary information
    context = augmented_context.split("\n\nSupplementary Information:")[0].split("Summary:")[0] 
    supplementary_info = augmented_context.split("\n\nSupplementary Information:")[1:][0]

    input_ = f"""
Given the following task description, patient context, relevant supplementary information, and PubMed abstracts, please make a prediction with reasoning based on the patient's context and relevant medical knowledge.

# Task #
{TASKS[TASK]["description"]} 
========================================

# Patient Context #
{context}
========================================

Supplementary Information:
{supplementary_info}
========================================

# PubMed Abstracts #
{"".join([f"{i+1}. {abstract}" for i, abstract in enumerate(abstracts)])}
========================================

Give the prediction and reasoning in the following format:
# Reasoning #  
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
"""

    response = get_claude_response(llm=MODEL, prompt=input_)

    return response

def process_patient(patient_id, base_contexts, augmented_contexts, patient_data, result):
    if MODE == "base":
        if patient_id not in result:
            base_context = base_contexts[patient_id]
            response = prompt_w_base_context(base_context)
            result[patient_id]["input"] = base_context
            result[patient_id]["ground_truth"] = patient_data[patient_id]["label"]
            result[patient_id]["reasoning_and_prediction"] = response
    elif MODE == "augmented":
        if patient_id not in result:
            augmented_context = augmented_contexts[patient_id]
            response = prompt_w_augmented_context(augmented_context)
            result[patient_id]["input"] = augmented_context
            result[patient_id]["ground_truth"] = patient_data[patient_id]["label"]
            result[patient_id]["reasoning_and_prediction"] = response

def main():
    context_dir = "/shared/eng/pj20/kelpie_exp_data/patient_context"
    ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_.pkl"
    base_context_path = f"{context_dir}/base_context/patient_contexts_{DATASET}_{TASK}.json"
    # reference_case_path = f"/shared/eng/pj20/kelpie_exp_data/reference_case/reference_cases_{DATASET}_{TASK}_0.json"
    patient_data_path =  f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}.json"
    test_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_samples_test.json"
    result_dir = "/shared/eng/pj20/kelpie_exp_data/results"
    
    sample_dataset = pickle.load(open(ehr_path, "rb"))
    base_contexts = json.load(open(base_context_path, "r"))
    patient_data = json.load(open(patient_data_path, "r"))
    test_sample = json.load(open(test_sample_path, "r"))
    # augmented_context_path = f"{context_dir}/augmented_context/patient_contexts_{DATASET}_{TASK}.json"
    # augmented_contexts = json.load(open(augmented_context_path, "r"))
    # reference_cases = json.load(open(reference_case_path, "r"))

    patient_id_train, patient_id_val, patient_id_test = set(), set(), set()
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
    for sample in train_dataset:
        patient_id_train.add(sample["patient_id"])
    for sample in val_dataset:
        patient_id_val.add(sample["patient_id"])  
    for sample in test_dataset:
        patient_id_test.add(sample["patient_id"])
        
    test_ids = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in test_sample]
            
    print(f'Number of test samples: {len(test_ids)}')
    
    
    if MODE == "base":
        print("Zero-shot prediction using base context...")
        result_base = defaultdict(dict)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for patient_id in test_ids:
                future = executor.submit(process_patient, patient_id, base_contexts, None, patient_data, result_base)
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

        with open(f"{result_dir}/{DATASET}_{TASK}_{MODEL}_base_zeroshot_test_results_multithreaded.json", "w") as f:
            json.dump(result_base, f, indent=4)
        
    # elif MODE == "augmented":
    #     print("Zero-shot prediction using augmented context...")
    #     result_augmented = defaultdict(dict)
            
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    #         futures = []  
    #         for patient_id in test_ids:
    #             future = executor.submit(process_patient, patient_id, None, augmented_contexts, patient_data, result_augmented)
    #             futures.append(future)

    #         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #             pass  
                    
    #     with open(f"{result_dir}/{DATASET}_{TASK}_{MODEL}_augmented_zeroshot_test_results_multithreaded.json", "w") as f:
    #         json.dump(result_augmented, f, indent=4)
        
        
    print("Zero-shot prediction completed.")
    

if __name__ == "__main__":
    main()