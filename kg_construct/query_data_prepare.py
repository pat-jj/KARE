import json
from collections import defaultdict
from itertools import chain
from tqdm import tqdm

TASKS = ["readmission", "mortality"]
DATASETS = ["mimic3", "mimic4"]
SAVE_DIR = "/shared/eng/pj20/kelpie_exp_data/kg_construct"

all_visit_concepts = []
all_concept_coexistence = defaultdict(lambda: defaultdict(int))

print("Processing datasets...")
for DATASET in DATASETS:
    for TASK in TASKS:
        agg_samples_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}.json"
        agg_samples = json.load(open(agg_samples_path, "r"))

        # Task 1: Get a list of sets of concepts for each visit
        visit_concepts = []
        for patient_id, patient_data in agg_samples.items():
            for visit_id, visit_data in patient_data.items():
                if visit_id.startswith("visit"):
                    visit_concepts.append(list(set(chain(visit_data["conditions"], visit_data["procedures"], visit_data["drugs"]))))

        # Task 2: Get top 20 co-existing concepts for each concept
        concept_coexistence = defaultdict(lambda: defaultdict(int))
        for visit_concept_set in visit_concepts:
            for concept1 in visit_concept_set:
                for concept2 in visit_concept_set:
                    if concept1 != concept2:
                        concept_coexistence[concept1][concept2] += 1
                        all_concept_coexistence[concept1][concept2] += 1

        top_coexisting_concepts = {}
        for concept, coexistence_counts in concept_coexistence.items():
            top_coexisting_concepts[concept] = [item[0] for item in sorted(coexistence_counts.items(), key=lambda x: x[1], reverse=True)[:20]]

        # Save the results as JSON
        with open(f"{SAVE_DIR}/{DATASET}_{TASK}_visit_concepts.json", "w") as f:
            json.dump(visit_concepts, f, indent=4)
        
        with open(f"{SAVE_DIR}/{DATASET}_{TASK}_top_coexisting_concepts.json", "w") as f:
            json.dump(top_coexisting_concepts, f, indent=4)

        all_visit_concepts.extend(visit_concepts)

# Aggregate results for task 2
all_top_coexisting_concepts = {}
for concept, coexistence_counts in all_concept_coexistence.items():
    all_top_coexisting_concepts[concept] = [item[0] for item in sorted(coexistence_counts.items(), key=lambda x: x[1], reverse=True)[:20]]

# Save the aggregate results as JSON
with open(f"{SAVE_DIR}/all_visit_concepts.json", "w") as f:
    json.dump(all_visit_concepts, f, indent=4)

with open(f"{SAVE_DIR}/all_top_coexisting_concepts.json", "w") as f:
    json.dump(all_top_coexisting_concepts, f, indent=4)