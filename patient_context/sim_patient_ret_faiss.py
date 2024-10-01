import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import faiss

MAX_K = 2  # Set to the maximum K needed

TASK  = 'readmission'
DATASET = 'mimic4'
MAX_CONTEXT_LENGTH = 20000

# File paths
PATIENT_CONTEXT_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{DATASET}_{TASK}_.json"
PATIENT_CONTEXT_PATH_SUB = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{DATASET}_{TASK}.json"
PATIENT_EMBEDDINGS_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_embeddings_{DATASET}_{TASK}.pkl"
PATIENT_DATA = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}_.json"

def is_context_valid(context):
    return len(context) <= MAX_CONTEXT_LENGTH

with open(PATIENT_CONTEXT_PATH, "r") as f:
    patient_contexts = json.load(f)
    
with open(PATIENT_CONTEXT_PATH_SUB, "r") as f:
    patient_contexts_sub = json.load(f)

with open(PATIENT_DATA, "r") as f:
    patient_data = json.load(f)
    
with open(PATIENT_EMBEDDINGS_PATH, "rb") as f:
    patient_embeddings = pickle.load(f)

patient_ids = list(patient_embeddings.keys())
patient_to_top_k_patient_contexts_k1 = {}
patient_to_top_k_patient_contexts_k2 = {}

print("Building Faiss index...")

# Prepare data for Faiss
embedding_matrix = np.array([patient_embeddings[pid] for pid in patient_ids]).astype('float32')

# Normalize embeddings
faiss.normalize_L2(embedding_matrix)

# Build the index
index = faiss.IndexFlatIP(embedding_matrix.shape[1])  # Inner product for normalized vectors is cosine similarity
index.add(embedding_matrix)

print("Creating patient ID to index mapping...")
patient_id_to_index = {pid: idx for idx, pid in enumerate(patient_ids)}

print("Searching for top-K similar patients...")

target_patient_ids = list(patient_contexts_sub.keys())

for patient_id in tqdm(target_patient_ids):
    if patient_id not in patient_id_to_index:
        print(f"Patient ID {patient_id} not found in embeddings.")
        continue  # Skip if patient_id is not in embeddings

    idx = patient_id_to_index[patient_id]
    if len(patient_contexts[patient_id]) > 30000:
        # For both K=1 and K=2, store "None"
        patient_to_top_k_patient_contexts_k1[patient_id] = {
            'positive': ["None"],
            'negative': ["None"]
        }
        patient_to_top_k_patient_contexts_k2[patient_id] = {
            'positive': ["None"],
            'negative': ["None"]
        }
        continue

    label = patient_data[patient_id]['label']
    current_embedding = embedding_matrix[idx].reshape(1, -1)

    # Search for the nearest neighbors
    D, I = index.search(current_embedding, 100)  # Search for more neighbors to filter later

    neighbor_ids = []
    for neighbor_idx in I[0]:
        neighbor_id = patient_ids[neighbor_idx]
        if neighbor_id == patient_id or neighbor_id.split("_")[0] == patient_id.split("_")[0]:
            continue  # Skip the same patient or other instances of the same patient
        neighbor_ids.append(neighbor_id)
        if len(neighbor_ids) >= 50:  # Collect enough neighbors to find top-K after filtering
            break

    # Compute similarity scores for the neighbors
    similarity_scores = {pid: np.dot(current_embedding[0], embedding_matrix[patient_id_to_index[pid]]) for pid in neighbor_ids}

    # Separate positive and negative patients
    positive_ids = [pid for pid in similarity_scores if patient_data[pid]['label'] == label]
    negative_ids = [pid for pid in similarity_scores if patient_data[pid]['label'] != label]

    # Sort and select top-K positive and negative patients
    sorted_positive_ids = sorted(
        positive_ids, key=lambda pid: similarity_scores[pid], reverse=True
    )
    sorted_positive_ids = [
        pid for pid in sorted_positive_ids if is_context_valid(patient_contexts[pid])
    ][:MAX_K]  # Get up to MAX_K positive patients

    sorted_negative_ids = sorted(
        negative_ids, key=lambda pid: similarity_scores[pid], reverse=True
    )
    sorted_negative_ids = [
        pid for pid in sorted_negative_ids if is_context_valid(patient_contexts[pid])
    ][:MAX_K]  # Get up to MAX_K negative patients

    # Prepare results for K=1
    patient_to_top_k_patient_contexts_k1[patient_id] = {
        'positive': [
            patient_contexts[pid] + f"\n\nLabel:\n{patient_data[pid]['label']}\n\n" for pid in sorted_positive_ids[:1]
        ] if sorted_positive_ids[:1] else ["None"],
        'negative': [
            patient_contexts[pid] + f"\n\nLabel:\n{patient_data[pid]['label']}\n\n" for pid in sorted_negative_ids[:1]
        ] if sorted_negative_ids[:1] else ["None"]
    }

    # Prepare results for K=2
    patient_to_top_k_patient_contexts_k2[patient_id] = {
        'positive': [
            patient_contexts[pid] + f"\n\nLabel:\n{patient_data[pid]['label']}\n\n" for pid in sorted_positive_ids[:2]
        ] if sorted_positive_ids[:2] else ["None"],
        'negative': [
            patient_contexts[pid] + f"\n\nLabel:\n{patient_data[pid]['label']}\n\n" for pid in sorted_negative_ids[:2]
        ] if sorted_negative_ids[:2] else ["None"]
    }

# Save the results to JSON files
output_path_k1 = f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_1_patient_contexts_{DATASET}_{TASK}.json"
output_path_k2 = f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_2_patient_contexts_{DATASET}_{TASK}.json"

with open(output_path_k1, "w") as f:
    json.dump(patient_to_top_k_patient_contexts_k1, f, indent=4)

with open(output_path_k2, "w") as f:
    json.dump(patient_to_top_k_patient_contexts_k2, f, indent=4)

print(f"Top-1 similar patient contexts saved to {output_path_k1}")
print(f"Top-2 similar patient contexts saved to {output_path_k2}")
