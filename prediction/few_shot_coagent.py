import json
import pickle
from spliter import split_by_patient
from apis.claude_api import get_claude_response
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import random
import os

DATASET = "mimic4"
TOP_K = 1
TASK = "mortality"

TASKS = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival

**Must to Notice:** Only the patients with extremely very high risk of mortality should be predicted as 1. 
"""
    },
    "readmission": {
        "description": """  
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days
"""
    },
}


INIT_GUIDELINES = {
    "mortality": """
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

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other.  
""",

    "readmission": """
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
}


def prompt_predictor(context, task, critic_feedback=None, similar_patient=None):
    similar_patients = similar_patient['positive'] + similar_patient['negative']
    random.shuffle(similar_patients)
    
    input_ = f"""
Given the following task description, patient EHR context, task instructions, and similar patients, please make a prediction with reasoning.

# Task #
{TASKS[task]["description"]}  

# Patient EHR Context #
{context}

# Task Instructions (Guidelines) #
{critic_feedback if critic_feedback else INIT_GUIDELINES[task]}

# Similar Patients #
{" ".join(similar_patients)}

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction # 
[Your prediction here (1/0)]

Output:
"""
    # print(input_)
    response = get_claude_response(llm="sonnet", prompt=input_)
    return input_, response



def prompt_critic(input_data_batch):
    input_ = f"""
You are an assistant who is good at self-reflection, gaining experience, and summarizing criteria. By reflecting on failure predictions that are given below, your task is to reflect on these incorrect predictions, compare them against the ground truth, and formulate criteria and guidelines to enhance the accuracy of future predictions.
The original instructions are provided under "# Task Instructions (Guidelines) #". Your task is to refine the instructions based on the discrepancies between the predictions and the ground truth.

# Input Data #
{input_data_batch}  

# Instructions #
1. Please always remember that the predictions above are all incorrect. You should always use the ground truth as the final basis to discover many unreasonable aspects in the predictions and then summarize them into experience and criteria.
2. Identify why the wrong predictions deviated from the ground truth by examining discrepancies in the medical history analysis.
3. Determine key and potential influencing factors, reasoning methods, and relevant feature combinations that could better align predictions with the ground truth.
4. The instructions should be listed in distinct rows, each representing a criteria or guideline.
5. The instructions should be generalizable to multiple samples, rather than specific to individual samples.
6. Conduct detailed analysis and write criteria based on the input samples, rather than writing some criteria without foundation.
7. Please note that the criteria you wrote should not include the word "ground truth".

Your output should be the new set of guidelines under "# Task Instructions (Guidelines) #" that can be used to improve the predictor's reasoning process.
Output **up to ten** guidelines themselves without any additional information.

Output:
"""
    response = get_claude_response(llm="sonnet", prompt=input_)
    return response

def process_patient(patient_id, base_contexts, patient_data, predictor_prompts, task, critic_feedback, similar_patients, result):
    # if patient_id not in result:
        base_context = base_contexts[patient_id]
        similar_patient = similar_patients[patient_id]
        input_, predictor_output = prompt_predictor(base_context, task, critic_feedback, similar_patient)
        
        predictor_prompts[patient_id] = input_
        result[patient_id]["input"] = input_
        result[patient_id]["prediction"] = predictor_output
        result[patient_id]["ground_truth"] = patient_data[patient_id]["label"]
    # else:
    #     print(f"Patient {patient_id} already processed.")

def consolidate_feedback(feedback_list):
    input_ = f"""
Given the following set of guidelines, please consolidate the insights into a concise and coherent set of guidelines for refining the predictor's reasoning process.

# Set of Guidelines #
{feedback_list}

# Instructions #
1. Analyze the provided guidelines and identify common themes, patterns, and key insights.
2. Synthesize the insights into a consolidated set of guidelines that capture the most important and recurring aspects.
3. Ensure that the consolidated guidelines are clear, concise, and actionable to refine the predictor's reasoning process.
4. Create a numbered list of the consolidated guidelines (**up to 15**) in the same format as the original guidelines.

Output:
"""
    response = get_claude_response(llm="sonnet", prompt=input_)
    return response

def consolidate_feedback_recursive(feedback_list):
    if len(feedback_list) <= 10:
        return consolidate_feedback(feedback_list)
    else:
        chunk_size = 5
        feedback_chunks = [feedback_list[i:i+chunk_size] for i in range(0, len(feedback_list), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as consolidate_executor:
            consolidate_futures = []
            for chunk in feedback_chunks:
                consolidate_future = consolidate_executor.submit(consolidate_feedback, chunk)
                consolidate_futures.append(consolidate_future)

            consolidated_feedback_list = [future.result() for future in tqdm(concurrent.futures.as_completed(consolidate_futures), total=len(consolidate_futures))]

        return consolidate_feedback_recursive(consolidated_feedback_list)
        
def main():
    context_dir = "/shared/eng/pj20/kelpie_exp_data/patient_context" 
    ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}.pkl"
    base_context_path = f"{context_dir}/base_context/patient_contexts_{DATASET}_{TASK}.json"
    similar_patient_path = f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_{TOP_K}_patient_contexts_{DATASET}_{TASK}.json"
    patient_data_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}.json"
    test_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_samples_test.json"
    result_dir = "/shared/eng/pj20/kelpie_exp_data/results"

    sample_dataset = pickle.load(open(ehr_path, "rb"))
    base_contexts = json.load(open(base_context_path, "r"))
    patient_data = json.load(open(patient_data_path, "r"))
    similar_patients = json.load(open(similar_patient_path, "r"))
    test_sample = json.load(open(test_sample_path, "r"))

    test_ids = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in test_sample]
    print(f'Number of test samples: {len(test_ids)}')

    # for task in ["mortality", "readmission"]:
    print(f"Performing {TASK} prediction...")
    result = defaultdict(dict)
    predictor_prompts = {}
    critic_feedback = None

    for i in range(4):  # Iterate for 3 rounds
        
        if os.path.exists(f"{result_dir}/{DATASET}_{TASK}_ehr_coagent_results_round_{i+2}.json"):
            print(f"Found existing results for round {i+2}. Skipping round {i+1}.")
            continue
        
        if os.path.exists(f"{result_dir}/{DATASET}_{TASK}_ehr_coagent_results_round_{i+1}.json"):
            with open(f"{result_dir}/{DATASET}_{TASK}_ehr_coagent_results_round_{i+1}.json", "r") as f:
                result = json.load(f)
            for pid in result.keys():
                predictor_prompts[pid] = result[pid]["input"]
        else:
            print(f"Round {i+1}:")
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = []
                print(f"Processing {len(test_ids)} patients...")
                for patient_id in test_ids:
                    future = executor.submit(process_patient, patient_id, base_contexts, patient_data, predictor_prompts, TASK, critic_feedback, similar_patients, result)
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
            with open(f"{result_dir}/{DATASET}_{TASK}_ehr_coagent_results_round_{i+1}.json", "w") as f:
                json.dump(result, f, indent=4)   
                    
                    

        incorrect_predictions = [(predictor_prompts[pid], result[pid]["prediction"], result[pid]["ground_truth"])
                                    for pid in result 
                                    if result[pid]["prediction"][-1] != str(result[pid]["ground_truth"])]
        
        batches = [incorrect_predictions[i:i+5] for i in range(0, len(incorrect_predictions), 5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as critic_executor:
            critic_futures = []
            for batch in batches:
                input_data_batch = [f"Prompt:\n{prompt}\n\nPrediction:\n{pred}\n\nGround Truth:\n{gt}\n" for prompt, pred, gt in batch]
                critic_future = critic_executor.submit(prompt_critic, "\n\n".join(input_data_batch))
                critic_futures.append(critic_future)

            critic_feedback_list = [future.result() for future in tqdm(concurrent.futures.as_completed(critic_futures), total=len(critic_futures))]

        # chunk_size = 5
        # feedback_chunks = [critic_feedback_list[i:i+chunk_size] for i in range(0, len(critic_feedback_list), chunk_size)]
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as consolidate_executor:
        #     consolidate_futures = []
        #     for chunk in feedback_chunks:
        #         consolidate_future = consolidate_executor.submit(consolidate_feedback, chunk)
        #         consolidate_futures.append(consolidate_future)

        #     consolidated_feedback_list = [future.result() for future in tqdm(concurrent.futures.as_completed(consolidate_futures), total=len(consolidate_futures))]

        # critic_feedback = consolidate_feedback(consolidated_feedback_list)
        critic_feedback = consolidate_feedback_recursive(critic_feedback_list)

        with open(f"{result_dir}/{DATASET}_{TASK}_ehr_coagent_results.json", "w") as f:
            json.dump(result, f, indent=4)

    print("EHR-CoAgent prediction completed.")

if __name__ == "__main__":
    main()