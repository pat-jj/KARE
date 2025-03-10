from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from utils import mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn, drug_recommendation_mimic3_fn, drug_recommendation_mimic4_fn
import csv
import pickle
from copy import deepcopy


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


def main():
    datasets = ["mimic3", 
                # "mimic4"
                ]
    tasks = [
        # "drugrec", 
        "mortality", 
        "readmission", 
        # "lenofstay"
        ]
    out_dir = "/shared/eng/pj20/kelpie_exp_data/baseline_data"
    
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
