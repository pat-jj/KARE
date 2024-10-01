import csv
from tqdm import tqdm
import os
import json
import random
from langdetect import detect


def read_code2name(file_path):
    condition_mapping_file = f"{file_path}/CCSCM.csv"
    procedure_mapping_file = f"{file_path}/CCSPROC.csv"
    drug_file = f"{file_path}/ATC.csv"

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

def read_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] not in mapping:
                mapping[row[0]] = []
            mapping[row[0]].append(row[1])
    return mapping

def read_umls_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in tqdm(reader):
            cui, term = row
            if cui not in mapping:
                mapping[cui] = []
            mapping[cui].append(term)
    return mapping

def read_ccs_to_icd9_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            icd9_code, ccs_code = row
            if ccs_code not in mapping:
                mapping[ccs_code] = icd9_code
    return mapping

def read_umls_graph(file_path):
    graph = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            if head not in graph:
                graph[head] = []
            graph[head].append((relation, tail))
    return graph

def extract_subgraph(graph, cui, depth=1):
    subgraph = []
    visited = set()
    queue = [(cui, 0)]
    while queue:
        node, d = queue.pop(0)
        if node in visited or d > depth:
            continue
        visited.add(node)
        if node in graph:
            for relation, tail in graph[node]:
                subgraph.append((node, relation, tail))
                if d < depth:
                    queue.append((tail, d + 1))
    return subgraph

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
    
def process_concept(umls_cui, umls_graph, umls_mapping, output_file):
    subgraph = extract_subgraph(umls_graph, umls_cui)
    unique_triples = set()
    if len(subgraph) > 200:
        subgraph = random.sample(subgraph, 200)
    for head, relation, tail in subgraph:
        if head in umls_mapping and tail in umls_mapping:
            head_terms = [term for term in umls_mapping[head] if is_english(term)]
            tail_terms = [term for term in umls_mapping[tail] if is_english(term)]
            if head_terms and tail_terms:
                unique_triples.add((head_terms[0], relation, tail_terms[0]))

    with open(output_file, 'a', encoding='utf-8') as f:
        for head, relation, tail in unique_triples:
            f.write(f"{head}\t{relation}\t{tail}\n")


def main():
    resource_path = "./resources"
    condition_dict, procedure_dict, drug_dict = read_code2name(resource_path)
    
    umls_graph_file = "/shared/eng/pj20/umls/graph.txt"
    umls_mapping_file = "/shared/eng/pj20/umls/UMLS.csv"
    atc_to_umls_file = "/home/pj20/server-04/Kelpie/kg_construct/umls_source/ATC_to_UMLS.csv"
    icd9_to_umls_file = "/home/pj20/server-04/Kelpie/kg_construct/umls_source/ICD9_to_UMLS.csv"
    icd9proc_to_ccsproc_file = "/home/pj20/server-04/Kelpie/kg_construct/umls_source/ICD9PROC_to_CCSPROC.csv"
    icd9cm_to_ccscm_file = "/home/pj20/server-04/Kelpie/kg_construct/umls_source/ICD9CM_to_CCSCM.csv"

    print("Reading UMLS graph...")
    umls_graph = read_umls_graph(umls_graph_file)
    print(f"UMLS graph loaded with {len(umls_graph)} nodes")

    print("Reading mappings...")
    umls_mapping = read_umls_mapping(umls_mapping_file)
    atc_to_umls = read_mapping(atc_to_umls_file)
    icd9_to_umls = read_mapping(icd9_to_umls_file)
    ccsproc_to_icd9proc = read_ccs_to_icd9_mapping(icd9proc_to_ccsproc_file)
    ccscm_to_icd9cm = read_ccs_to_icd9_mapping(icd9cm_to_ccscm_file)
    print("Mappings loaded")

    graph_path = "./graphs"
    os.makedirs(graph_path, exist_ok=True)

    # Extract subgraphs for ATC-3 codes
    print("Extracting subgraphs for ATC-3 codes...")
    mode_path = os.path.join(graph_path, "drug", "UMLS")
    os.makedirs(mode_path, exist_ok=True)
    for atc_code in tqdm(drug_dict.keys()):
        if atc_code in atc_to_umls:
            umls_cuis = atc_to_umls[atc_code]
            output_file = os.path.join(mode_path, f"{atc_code}.txt")
            for umls_cui in umls_cuis:
                process_concept(umls_cui, umls_graph, umls_mapping, output_file)

    # Extract subgraphs for CCSCM codes
    print("Extracting subgraphs for CCSCM codes...")
    mode_path = os.path.join(graph_path, "condition", "UMLS")
    os.makedirs(mode_path, exist_ok=True)
    for ccscm_code in tqdm(condition_dict.keys()):
        try:
            icd9cm_code = ccscm_to_icd9cm[ccscm_code]
        except:
            continue
        if icd9cm_code in icd9_to_umls:
            umls_cuis = icd9_to_umls[icd9cm_code]
            output_file = os.path.join(mode_path, f"{ccscm_code}.txt")
            for umls_cui in umls_cuis:
                process_concept(umls_cui, umls_graph, umls_mapping, output_file)

    # Extract subgraphs for CCSPROC codes
    print("Extracting subgraphs for CCSPROC codes...")
    mode_path = os.path.join(graph_path, "procedure", "UMLS")
    os.makedirs(mode_path, exist_ok=True)
    for ccsproc_code in tqdm(procedure_dict.keys()):
        try:
            icd9proc_code = ccsproc_to_icd9proc[ccsproc_code]
        except:
            continue
        if icd9proc_code in icd9_to_umls:
            umls_cuis = icd9_to_umls[icd9proc_code]
            output_file = os.path.join(mode_path, f"{ccsproc_code}.txt")
            for umls_cui in umls_cuis:
                process_concept(umls_cui, umls_graph, umls_mapping, output_file)

if __name__ == "__main__":
    main()