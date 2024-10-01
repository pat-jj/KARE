import pickle
from spliter import split_by_patient
import json
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import random
import torch


# Load patient embeddings
with open("/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/patient_embeddings.pkl", "rb") as f:
    patient_embeddings = pickle.load(f)
    
# Load knowledge embeddings
with open("/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/knowledge_embeddings.pkl", "rb") as f:
    knowledge_embeddings = pickle.load(f)
    
# Load patient knowledge nodes
with open("/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/parsed_data_iteration_0.json", "r") as f:
    patient_knowledge_node = json.load(f)

# Load patient contexts
context_dir = "/shared/eng/pj20/kelpie_exp_data/patient_context/augmented_context/patient_contexts_mimic3_readmission.json"
with open(context_dir, "r") as f:
    patient_contexts = json.load(f)

# Clean patient contexts and create context embeddings dictionary
context_embeddings = {}
for patient_id, context in patient_contexts.items():
    cleaned_context = context.split("\n\nSupplementary Information:")[0].split("Summary:")[0]
    context_embeddings[cleaned_context] = patient_embeddings[patient_id]


# Load EHR data
ehr_path = "/shared/eng/pj20/kelpie_exp_data/ehr_data/mimic3_readmission.pkl"
with open(ehr_path, "rb") as f:
    sample_dataset = pickle.load(f)

# Split the dataset
train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)

# Collect patient IDs for each split
patient_id_train = {sample["patient_id"] for sample in train_dataset}
patient_id_val = {sample["patient_id"] for sample in val_dataset}
patient_id_test = {sample["patient_id"] for sample in test_dataset}


# Create mappings from knowledge IDs to texts
knowledge_id_to_text = {knowledge_id: knowledge_id for knowledge_id in knowledge_embeddings.keys()}

# Prepare training examples
train_examples = []

for patient_id in patient_contexts.keys():
    if patient_id.split("_")[0] not in patient_id_train:
        continue
    # Get and clean patient context
    context = patient_contexts[patient_id]
    context = context.split("\n\nSupplementary Information:")[0].split("Summary:")[0]
    
    # Get positive knowledge IDs and texts
    positive_knowledge_ids = list(patient_knowledge_node[patient_id].keys())
    positive_knowledge_texts = [knowledge_id_to_text[kid] for kid in positive_knowledge_ids]
    
    # Create positive examples
    for pos_text in positive_knowledge_texts:
        train_examples.append(InputExample(texts=[context, pos_text], label=1.0))
    
    # Get negative knowledge IDs and texts
    all_knowledge_ids = set(knowledge_embeddings.keys())
    negative_knowledge_ids = all_knowledge_ids - set(positive_knowledge_ids)
    negative_knowledge_sample = random.sample(negative_knowledge_ids, k=min(10, len(negative_knowledge_ids)))
    negative_knowledge_texts = [knowledge_id_to_text[kid] for kid in negative_knowledge_sample]
    
    # Create negative examples
    for neg_text in negative_knowledge_texts:
        train_examples.append(InputExample(texts=[context, neg_text], label=0.0))
        
        
# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define a loss function
train_loss = losses.CosineSimilarityLoss(model=model)


# Define the number of epochs and warmup steps
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/trained_dense_retriever'
)


# Encode knowledge texts
knowledge_texts = [knowledge_id_to_text[kid] for kid in knowledge_embeddings.keys()]
knowledge_corpus_embeddings = model.encode(knowledge_texts, convert_to_tensor=True)

def retrieve_top_knowledge(model, patient_context: str, knowledge_corpus_embeddings, knowledge_texts, top_k: int = 10):
    # Encode the patient context
    query_embedding = model.encode(patient_context, convert_to_tensor=True)
    
    # Perform semantic search
    hits = util.semantic_search(query_embedding, knowledge_corpus_embeddings, top_k=top_k)[0]
    
    # Retrieve the top knowledge texts
    top_knowledge = [knowledge_texts[hit['corpus_id']] for hit in hits]
    return top_knowledge



# Load the trained model (if not already loaded)
model = SentenceTransformer('/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/trained_dense_retriever')

# Example patient context
patient_context = patient_contexts["10117_0"].split("\n\nSupplementary Information:")[0].split("Summary:")[0]

# Retrieve top 10 knowledge items
top_10_knowledge = retrieve_top_knowledge(model, patient_context, knowledge_corpus_embeddings, knowledge_texts)

# Print the results
print(f"Top 10 knowledge for patient context: {patient_context}")
print(top_10_knowledge)


