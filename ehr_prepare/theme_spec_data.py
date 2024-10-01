from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from utils import mortality_prediction_mimic3_fn_lung, lung_cancer_prediction_mimic3_fn
import csv
import pickle
from copy import deepcopy
import os

LUNG_CANCER_CODES = [162.2, 162.3, 162.4, 162.5, 162.8, 162.9]

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
        # refresh_cache=True,
        )
    elif dataset == "mimic4":
        ds = MIMIC4Dataset(
        root="/shared/eng/pj20/mimiciv/2.0/hosp/", 
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],      
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD10CM": "ICD9CM",
            "ICD10PROC": "ICD9PROC",
            },
        dev=False
        )
        
    return ds


def assign_task(dataset, ds, task):

    if task == "lung_mortality":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(mortality_prediction_mimic3_fn_lung)
            
    elif task == "lung_cancer":
        if dataset == "mimic3":
            sample_dataset = ds.set_task(lung_cancer_prediction_mimic3_fn)

    
    return sample_dataset


def main():
    datasets = [
        "mimic3", 
                # "mimic4"
                ]
    tasks = [
        # "drugrec", 
        "lung_mortality", 
        "lung_cancer", 
        # "lenofstay"
        ]
    out_dir = "/shared/eng/pj20/kelpie_exp_data/ehr_data"
    

    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        ds = load_dataset(dataset)
        for task in tasks:
            base_ds = deepcopy(ds)
            print(f"Assigning task: {task}")
            sample_dataset = assign_task(dataset, base_ds, task)
            print(f"Dataset: {dataset}, Task: {task}, Number of samples: {len(sample_dataset)}")
            print(f"Saving dataset to {out_dir}/{dataset}_{task}.pkl")
            with open(f"{out_dir}/{dataset}_{task}.pkl", "wb") as f:
                pickle.dump(sample_dataset, f)
    
                
    print("Done!")
    
    
if __name__ == "__main__":
    main()