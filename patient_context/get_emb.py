from apis.gpt_emb_api import get_embedding
import pickle
from tqdm import tqdm
import json
import concurrent.futures

TASK  = 'mortality_'
DATASET = 'mimic4'
# File paths
PATIENT_CONTEXT_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{DATASET}_{TASK}.json"
PATIENT_EMBEDDINGS_PATH = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_embeddings_{DATASET}_{TASK}.pkl"

# Number of threads; adjust based on your system and API rate limits
MAX_WORKERS = 15

# Maximum number of characters allowed for embedding
MAX_CHAR_LENGTH = 30000 # Adjust as needed based on token approximation

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def truncate_text(text, max_length):
    """Truncate text to a maximum number of characters."""
    if len(text) > max_length:
        truncated = text[:max_length]
        return truncated, True
    return text, False

def get_patient_embedding(patient_id, patient_info):
    """Retrieve embedding for a single patient, truncating if necessary."""
    text = str(patient_info)
    truncated_text, was_truncated = truncate_text(text, MAX_CHAR_LENGTH)
    
    if was_truncated:
        print(f"Truncated patient_id {patient_id} to {MAX_CHAR_LENGTH} characters.")
    
    try:
        embedding = get_embedding(truncated_text)
        return (patient_id, embedding)
    except Exception as exc:
        print(f"Error retrieving embedding for {patient_id}: {exc}")
        return (patient_id, None)

def get_knowledge_embedding(knowledge):
    """Retrieve embedding for a single knowledge item."""
    try:
        embedding = get_embedding(knowledge)
        return (knowledge, embedding)
    except Exception as exc:
        print(f"Error retrieving embedding for knowledge item '{knowledge}': {exc}")
        return (knowledge, None)

def retrieve_embeddings_multithread(data_dict, retrieval_func, description):
    """
    Retrieve embeddings using multithreading.

    Args:
        data_dict (dict): Dictionary of data items to process.
        retrieval_func (callable): Function to retrieve embedding.
        description (str): Description for the progress bar.

    Returns:
        dict: Dictionary mapping keys to their embeddings.
    """
    embeddings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Prepare the tasks
        if retrieval_func.__code__.co_argcount == 2:
            # Function expects key and value
            future_to_key = {
                executor.submit(retrieval_func, key, value): key
                for key, value in data_dict.items()
            }
        else:
            # Function expects only key
            future_to_key = {
                executor.submit(retrieval_func, key): key
                for key in data_dict.keys()
            }

        # Iterate over the completed futures as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_key),
                           total=len(future_to_key),
                           desc=description):
            key = future_to_key[future]
            try:
                result = future.result()
                if result is not None:
                    result_key, embedding = result
                    if embedding is not None:
                        embeddings[result_key] = embedding
            except Exception as exc:
                print(f"Error retrieving embedding for {key}: {exc}")
    return embeddings

def main():
    # Load data
    print("Loading data...")
    patient_data = load_json(PATIENT_CONTEXT_PATH)
    print("Data loaded successfully.")

    # Retrieve patient embeddings
    print("Retrieving patient embeddings...")
    patient_embeddings = retrieve_embeddings_multithread(
        data_dict=patient_data,
        retrieval_func=get_patient_embedding,
        description="Patients"
    )
    save_pickle(patient_embeddings, PATIENT_EMBEDDINGS_PATH)
    print(f"Patient embeddings saved to {PATIENT_EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()
