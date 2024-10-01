from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from utils import mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn, drug_recommendation_mimic3_fn, drug_recommendation_mimic4_fn
import csv
import pickle
from copy import deepcopy
from collections import defaultdict
import json
import random
from pyhealth.datasets import SampleDataset


def load_mappings():
    condition_mapping_file = "./resources/CCSCM.csv"
    procedure_mapping_file = "./resources/CCSPROC.csv"
    drug_file = "./resources/ATC.csv"

    condition_dict = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()

    procedure_dict = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()

    drug_dict = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()

    return condition_dict, procedure_dict, drug_dict


def load_dataset(dataset):
    if dataset == "mimic3":
        ds = MIMIC3Dataset(
        root="/shared/eng/pj20/mimiciii/1.4/", 
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],      
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC"
            },
        dev=False,
        refresh_cache=False,
        )
    elif dataset == "mimic4":
        ds = MIMIC4Dataset(
        root="/shared/eng/pj20/mimiciv/2.0/hosp/", 
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],      
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM",
            "ICD10PROC": "CCSPROC",
            },
        dev=False,
        refresh_cache=True,
        )
        
    return ds
        

def assign_task(dataset, ds, task):

    if task == "drugrec":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(drug_recommendation_mimic3_fn)
        if dataset == "mimic4":
            sample_dataset = ds.set_task(drug_recommendation_mimic4_fn)
    elif task == "mortality":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(mortality_prediction_mimic3_fn)
        if dataset == "mimic4":
            sample_dataset = ds.set_task(mortality_prediction_mimic4_fn)
    elif task == "readmission":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(readmission_prediction_mimic3_fn)
        if dataset == "mimic4":
            sample_dataset = ds.set_task(readmission_prediction_mimic4_fn)
    elif task == "lenofstay":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(length_of_stay_prediction_mimic3_fn)
        elif dataset == "mimic4":
            sample_dataset = ds.set_task(length_of_stay_prediction_mimic4_fn)
    
    return sample_dataset

def expand_and_map(l, dict_):
    if type(l[0]) == list:
        return [dict_[item] for sublist in l for item in sublist]
    if type(l[0]) == str:
        return [dict_[item] for item in l]
    
def process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict):
    patient_data = defaultdict(dict)
    patient_data_no_label = defaultdict(dict)
    patient_to_index = sample_dataset.patient_to_index
    
    for patient, idxs in patient_to_index.items():
        for i in range(len(idxs)):
            label = sample_dataset.samples[idxs[i]]['label']
            patient_id = patient + f"_{i}"
            patient_data[patient_id]['label'] = label
            for j in range(i+1):
                idx = idxs[j]
                data = sample_dataset.samples[idx]
                conditions = expand_and_map(data['conditions'], condition_dict)
                procedures = expand_and_map(data['procedures'], procedure_dict)
                drugs = expand_and_map(data['drugs'], drug_dict)
                patient_data[patient_id][f'visit {j}'] = {
                    'conditions': conditions,
                    'procedures': procedures,
                    'drugs': drugs
                }
                patient_data_no_label[patient_id][f'visit {j}'] = {
                    'conditions': conditions,
                    'procedures': procedures,
                    'drugs': drugs   
                }
            
    return patient_data, patient_data_no_label


def main():
    datasets = [
        # "mimic3", 
            "mimic4"
                ]
    tasks = [
        # "drugrec", 
        "mortality", 
        "readmission", 
        # "lenofstay"
        ]
    out_dir = "/shared/eng/pj20/kelpie_exp_data/ehr_data"
    condition_dict, procedure_dict, drug_dict = load_mappings()
    
    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        ds = load_dataset(dataset)
        for task in tasks:
            base_ds = deepcopy(ds)
            print(f"Assigning task: {task}")
            sample_dataset = assign_task(dataset, base_ds, task)
            print(f"Dataset: {dataset}, Task: {task}, Number of samples: {len(sample_dataset)}")
            print(f"Saving dataset to {out_dir}/{dataset}_{task}.pkl")
            if dataset == "mimic3":
                with open(f"{out_dir}/{dataset}_{task}.pkl", "wb") as f:
                    pickle.dump(sample_dataset, f)
            elif dataset == "mimic4":
                with open(f"{out_dir}/{dataset}_{task}_.pkl", "wb") as f:
                    pickle.dump(sample_dataset, f)
                
            sample_dataset_path = f"{out_dir}/{dataset}_{task}.pkl"
            sample_dataset = pickle.load(open(sample_dataset_path, "rb"))
            patient_data, patient_data_no_label = process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict)
            if dataset == "mimic3":
                with open(f"{out_dir}/pateint_{dataset}_{task}.json", "w") as f:
                    json.dump(patient_data, f, indent=4)
            elif dataset == "mimic4":
                with open(f"{out_dir}/pateint_{dataset}_{task}_.json", "w") as f:
                    json.dump(patient_data, f, indent=4)
                
    print("Done!")
    
    
    # mimic-4 specific processing
    for task in tasks:
        with open(f"{out_dir}/pateint_mimic4_{task}_.json", "r") as f:
            data = json.load(f)
            
        mimic4_samples = defaultdict(dict)

        random.seed(42)
        random.shuffle(list(data.keys()))

        patient_set = set()

        label_1_cnt = 0

        for patient_id in data.keys():
            if data[patient_id]['label'] == 1 and patient_id.split("_")[0] not in patient_set:
                if len(data[patient_id]) <= 10:
                    mimic4_samples[patient_id] = data[patient_id]
                    label_1_cnt += 1
                    patient_set.add(patient_id.split("_")[0])
                if len(mimic4_samples) == 5000:
                    break
                

        label_0_cnt = 0
        for i, patient_id in enumerate(data.keys()):
            if data[patient_id]['label'] == 0 and patient_id.split("_")[0] not in patient_set:
                if len(data[patient_id]) <= 10:
                    mimic4_samples[patient_id] = data[patient_id]
                    label_0_cnt += 1
                    patient_set.add(patient_id.split("_")[0])
                
            if len(mimic4_samples) == 10000:
                break
            

        with open(f"{out_dir}/pateint_mimic4_{task}.json" , "w") as f:
            json.dump(mimic4_samples, f, indent=4)
            
        print(f"Label 1: {label_1_cnt}")
        print(f"Label 0: {label_0_cnt}")
        
        with open(f"{out_dir}/pateint_mimic4_{task}.json", "r") as f:
            data = json.load(f)

        with open(f"{out_dir}/mimic4_{task}_.pkl", "rb") as f:
            mimic4_sample_dataset = pickle.load(f)
            
        patient_ids = list(data.keys())
        pat_ori_id = []

        for id_ in patient_ids:
            pat_ori_id.append(id_.split("_")[0])
            
        new_sample_list = []

        for sample in mimic4_sample_dataset.samples:
            if sample['patient_id'] in pat_ori_id:
                new_sample_list.append(sample)
                
        mimic4_sample_dataset = SampleDataset(new_sample_list)

        with open(f"{out_dir}/mimic4_{task}.pkl", "wb") as f:
            pickle.dump(mimic4_sample_dataset, f)
    
        
    
if __name__ == "__main__":
    main()
