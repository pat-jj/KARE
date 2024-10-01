from spliter import split_by_patient
import pickle
import json
from tqdm import tqdm

DATASET = "mimic4"
TASK = 'readmission'

ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_.pkl"
agg_samples_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{DATASET}_{TASK}.json"
sample_dataset = pickle.load(open(ehr_path, "rb"))
agg_samples = json.load(open(agg_samples_path, "r"))

patient_id_train, patient_id_val, patient_id_test = set(), set(), set()

train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
for sample in train_dataset:
    patient_id_train.add(sample["patient_id"])
for sample in val_dataset:
    patient_id_val.add(sample["patient_id"])
for sample in test_dataset:
    patient_id_test.add(sample["patient_id"])
    
    
# {"visit_id": "161106", "patient_id": "10004", "conditions": ["231", "233", "83", "130", "101", "49", "2603", "155"], "procedures": ["158", "3"], "drugs": ["B05X", "A12B", "N05B", "N02A", "A12C", "V04C", "A02B", "J01D", "N03A", "C07A", "N02B", "R03A", "D06A"], "label": 0}
samples_ehr_train, samples_ehr_val, samples_ehr_test = [], [], []
for patient_id in tqdm(agg_samples.keys()):
    conditions_all, procedures_all, drugs_all = [], [], []
    for i in range(len(agg_samples[patient_id])-1):
        conditions_all.append(agg_samples[patient_id][f'visit {i}']['conditions'])
        procedures_all.append(agg_samples[patient_id][f'visit {i}']['procedures'])
        drugs_all.append(agg_samples[patient_id][f'visit {i}']['drugs'])
    
    sample_new = {
        "visit_id": f"{patient_id.split('_')[1]}",
        'patient_id': patient_id.split('_')[0],
        "conditions": conditions_all,
        "procedures": procedures_all,
        "drugs": drugs_all,
        "label": agg_samples[patient_id]['label']
    }
    
    if patient_id.split('_')[0] in patient_id_train:
        samples_ehr_train.append(sample_new)
    elif patient_id.split('_')[0] in patient_id_val:
        samples_ehr_val.append(sample_new)
    elif patient_id.split('_')[0] in patient_id_test:
        samples_ehr_test.append(sample_new)
        
with open(f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_samples_train.json", "w") as f:
    json.dump(samples_ehr_train, f)

with open(f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_samples_val.json", "w") as f:
    json.dump(samples_ehr_val, f)
    
with open(f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{DATASET}_{TASK}_samples_test.json", "w") as f:
    json.dump(samples_ehr_test, f)