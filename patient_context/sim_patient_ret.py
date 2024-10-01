import os
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from spliter import split_by_patient

TOP_K = 1

TASK  = 'mortality'
DATASET = 'mimic4'
MAX_CONTEXT_LENGTH = 20000
# File paths
PATIENT_CONTEXT_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{DATASET}_{TASK}_.json"
PATIENT_EMBEDDINGS_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_embeddings_{DATASET}_{TASK}.pkl"
PATIENT_DATA = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}_.json"
# EHR_PATH = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}.pkl"

def is_context_valid(context):
    return len(context) <= MAX_CONTEXT_LENGTH

# sample_dataset = pickle.load(open(EHR_PATH, "rb"))
# patient_id_train, patient_id_val, patient_id_test = set(), set(), set()
# train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
# for sample in train_dataset:
#     patient_id_train.add(sample["patient_id"])
# for sample in val_dataset:
#     patient_id_val.add(sample["patient_id"])
# for sample in test_dataset:
#     patient_id_test.add(sample["patient_id"])

with open(PATIENT_CONTEXT_PATH, "r") as f:
    patient_contexts = json.load(f)

with open(PATIENT_DATA, "r") as f:
    patient_data = json.load(f)
    
with open(PATIENT_EMBEDDINGS_PATH, "rb") as f:
    patient_embeddings = pickle.load(f)
    
patient_to_top_k_patient_contexts = {}

print("Constructing patient similarity matrix...")
# Convert patient embeddings to a matrix
patient_ids = list(patient_embeddings.keys())
#exclude the patient from the test set
# patient_ids = [pid for pid in patient_ids if pid.split("_")[0] not in patient_id_test]

embedding_matrix = np.array([patient_embeddings[patient_id] for patient_id in patient_ids])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(embedding_matrix)
print("Done!")

for idx, patient_id in tqdm(enumerate(patient_ids)):
    label = patient_data[patient_id]['label']
    
    # Get similarity scores for the current patient
    similarity_scores = similarity_matrix[idx]
    
    # Create masks for positive and negative pools
    positive_mask = np.array([patient_data[p_id]['label'] == label for p_id in patient_ids])
    negative_mask = np.array([patient_data[p_id]['label'] != label for p_id in patient_ids])
    
    # Create a mask to exclude the current patient and any other instances of the same patient
    current_patient_mask = np.array([p_id.split("_")[0] != patient_id.split("_")[0] for p_id in patient_ids])
    
    # Combine the masks
    positive_mask = positive_mask & current_patient_mask
    negative_mask = negative_mask & current_patient_mask
    
    # Get top K valid contexts from positive and negative pools
    top_k_positive_ids = []
    top_k_negative_ids = []

    for i in np.argsort(similarity_scores * positive_mask)[::-1]:
        if positive_mask[i] and is_context_valid(patient_contexts[patient_ids[i]]):
            top_k_positive_ids.append(patient_ids[i])
            if len(top_k_positive_ids) == TOP_K:
                break

    for i in np.argsort(similarity_scores * negative_mask)[::-1]:
        if negative_mask[i] and is_context_valid(patient_contexts[patient_ids[i]]):
            top_k_negative_ids.append(patient_ids[i])
            if len(top_k_negative_ids) == TOP_K:
                break
    
    patient_to_top_k_patient_contexts[patient_id] = {
        'positive': [patient_contexts[p_id] + f"\n\nLabel:\n{patient_data[p_id]['label']}\n\n" for p_id in top_k_positive_ids],
        'negative': [patient_contexts[p_id] + f"\n\nLabel:\n{patient_data[p_id]['label']}\n\n" for p_id in top_k_negative_ids]
    }
# Save the results to a JSON file
with open(f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_{TOP_K}_patient_contexts_{DATASET}_{TASK}.json", "w") as f:
    json.dump(patient_to_top_k_patient_contexts, f, indent=4)