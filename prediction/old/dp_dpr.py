# import pickle
# from spliter import split_by_patient
# import json
# from sentence_transformers import SentenceTransformer, InputExample, losses, util
# from torch.utils.data import DataLoader
# import random
# import torch

# def retrieve_top_knowledge(model, patient_context: str, knowledge_corpus_embeddings, knowledge_texts, top_k: int = 10):
#     # Encode the patient context
#     query_embedding = model.encode(patient_context, convert_to_tensor=True)
    
#     # Perform semantic search
#     hits = util.semantic_search(query_embedding, knowledge_corpus_embeddings, top_k=top_k)[0]
    
#     # Retrieve the top knowledge texts
#     top_knowledge = [knowledge_texts[hit['corpus_id']] for hit in hits]
#     return top_knowledge


# def format_knowledge(knowledge):
#     formatted_knowledge = \
# f"""[
# 1. {knowledge[0]}

# 2. {knowledge[1]}

# 3. {knowledge[2]}

# 4. {knowledge[3]}

# 5. {knowledge[4]}

# 6. {knowledge[5]}

# 7. {knowledge[6]}

# 8. {knowledge[7]}

# 9. {knowledge[8]}

# 10. {knowledge[9]}
# ]

# """
#     return formatted_knowledge

# # Load the trained model (if not already loaded)
# model = SentenceTransformer('/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/trained_dense_retriever')

# print("Model loaded!")

# # Load knowledge embeddings
# with open("/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/knowledge_embeddings.pkl", "rb") as f:
#     knowledge_embeddings = pickle.load(f)
# # Create mappings from knowledge IDs to texts
# knowledge_id_to_text = {knowledge_id: knowledge_id for knowledge_id in knowledge_embeddings.keys()}

# # Encode knowledge texts
# knowledge_texts = [knowledge_id_to_text[kid] for kid in knowledge_embeddings.keys()]
# knowledge_corpus_embeddings = model.encode(knowledge_texts, convert_to_tensor=True)

# print("Knowledge embeddings loaded!")



# # Load EHR data
# ehr_path = "/shared/eng/pj20/kelpie_exp_data/ehr_data/mimic3_readmission.pkl"
# with open(ehr_path, "rb") as f:
#     sample_dataset = pickle.load(f)

# # Split the dataset
# train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)

# # Collect patient IDs for each split
# patient_id_test = {sample["patient_id"] for sample in test_dataset}


# # Load patient contexts
# context_dir = "/shared/eng/pj20/kelpie_exp_data/patient_context/augmented_context/patient_contexts_mimic3_readmission.json"
# with open(context_dir, "r") as f:
#     patient_contexts = json.load(f)

# original_file = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/mimic3_readmission_test_0.jsonl"
# data = []
# with open(original_file, "r") as f:
#     for line in f:
#         data.append(json.loads(line))

# save_path = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_dp/mimic3_readmission_new_test_dpr.jsonl"


# print("Starting to retrieve knowledge...")
# for i in range(len(data)):
#     sample = data[i]
#     original_input = sample["input"]
#     patient_id = original_input.split("Patient ID: ")[1].split("\n")[0]
#     context = patient_contexts[patient_id]
#     context = context.split("\n\nSupplementary Information:")[0].split("Summary:")[0]
    
#     # Retrieve top 10 knowledge items
#     top_10_knowledge = retrieve_top_knowledge(model, context, knowledge_corpus_embeddings, knowledge_texts)
    
#     # Format the knowledge
#     formatted_knowledge = format_knowledge(top_10_knowledge)

#     pre_text = original_input.split("\n# Retrieved Medical Knowledge #\n\n")[0]
#     new_input = pre_text + "\n# Retrieved Medical Knowledge #\n\n" + formatted_knowledge
#     data[i]["input"] = new_input
    
# with open(save_path, "w") as f:
#     for item in data:
#         f.write(json.dumps(item) + "\n")



