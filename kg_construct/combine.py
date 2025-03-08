import os
from collections import defaultdict
from typing import List, Tuple
import json
import csv
from tqdm import tqdm


REL_MAPPING = {
    "AQ": "Allowed qualifier",
    "CHD": "has child",
    "DEL": "Deleted concept",
    "PAR": "has parent",
    "QB": "can be qualified by.",
    "RB": "has a broader relationship",
    "RL": "alike",
    "RN": "has a narrower relationship",
    "RO": "has relationship",
    "RQ": "related and possibly synonymous.",
    "RU": "Related, unspecified",
    "SY": "source asserted synonymy.",
    "XR": "Not related, no mapping",
    "": "Empty relationship"
}


def load_mappings():
    condition_mapping_file = "./resources/CCSCM.csv"
    procedure_mapping_file = "./resources/CCSPROC.csv"
    drug_file = "./resources/ATC.csv"

    condition_dict = {}
    condition_dict_inv = {}
    with open(condition_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()
            condition_dict_inv[row['name'].lower()] = row['code']

    procedure_dict = {}
    procedure_dict_inv = {}
    with open(procedure_mapping_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()
            procedure_dict_inv[row['name'].lower()] = row['code']

    drug_dict = {}
    drug_dict_inv = {}
    with open(drug_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()
                drug_dict_inv[row['name'].lower()] = row['code']

    return condition_dict, procedure_dict, drug_dict, condition_dict_inv, procedure_dict_inv, drug_dict_inv


def combine_knowledge_graphs(graph_path: str, extracted_kg_path: str) -> None:
    
    all_triples = ""
    # from kg_from_kg.json
    kg_from_kg = json.load(open(os.path.join(extracted_kg_path, "kg_from_kg.json"), "r"))
    for key, value in kg_from_kg.items():
        for k, v in value.items():
            triples = v["triples"]
            for triple in triples:
                head, relation, tail = triple
                if relation in REL_MAPPING:
                    relation = REL_MAPPING[relation]
                all_triples += f"{head}\t{relation}\t{tail}\n"
    
    # from kg_from_llm.json
    kg_from_llm = json.load(open(os.path.join(extracted_kg_path, "kg_from_llm.json"), "r"))
    for triple in kg_from_llm:
        head, relation, tail = triple
        if "'" == head[0] or "'" == head[-1]:
            head = head[1:-1]
        if "'" == relation[0] or "'" == relation[-1]:
            relation = relation[1:-1]
        if "'" == tail[0] or "'" == tail[-1]:
            tail = tail[1:-1]
        all_triples += f"{head}\t{relation}\t{tail}\n"
    
    # kg_from_pubmed.json
    kg_from_pubmed = json.load(open(os.path.join(extracted_kg_path, "kg_from_pubmed.json"), "r"))
    for value, triples in kg_from_pubmed.items():
        for triple in triples:
            head, relation, tail = triple
            if "'" == head[0] or "'" == head[-1]:
                head = head[1:-1]
            if "'" == relation[0] or "'" == relation[-1]:
                relation = relation[1:-1]
            if "'" == tail[0] or "'" == tail[-1]:
                tail = tail[1:-1]
            all_triples += f"{head}\t{relation}\t{tail}\n"
            
    # save all_triples to a file
    with open(os.path.join(graph_path, "kg_raw.txt"), "w") as f:
        f.write(all_triples)
    
    
def read_from_extracted_kg(concept, extracted_kg_path):
    concept_triples = ""
    cnt = 0
    
    # kg_from_kg.json
    kg_from_kg = json.load(open(os.path.join(extracted_kg_path, "kg_from_kg.json"), "r"))
    if concept in kg_from_kg.keys():
        for k, v in kg_from_kg[concept].items():
            triples = v["triples"]
            for triple in triples:
                head, relation, tail = triple
                if relation in REL_MAPPING:
                    relation = REL_MAPPING[relation]
                concept_triples += f"{head}\t{relation}\t{tail}\n"
                cnt += 1
                
    # Process kg_from_llm for 2-hop subgraph
    kg_from_llm = json.load(open(os.path.join(extracted_kg_path, "kg_from_llm.json"), "r"))
    # First collect all directly connected entities (1-hop)
    connected_entities = set()
    entity_frequency = defaultdict(int)  # Track frequency of connected entities
    first_hop_triples = []
    
    for triple in kg_from_llm:
        head, relation, tail = triple
        # Clean up quotes
        head = head[1:-1] if "'" == head[0] or "'" == head[-1] else head
        relation = relation[1:-1] if "'" == relation[0] or "'" == relation[-1] else relation
        tail = tail[1:-1] if "'" == tail[0] or "'" == tail[-1] else tail
        
        # If concept is in the triple, add it and track connected entities
        if concept.lower() in [head.lower(), tail.lower()]:
            concept_triples += f"{head}\t{relation}\t{tail}\n"
            cnt += 1
            if head.lower() != concept.lower():
                connected_entities.add(head.lower())
                entity_frequency[head.lower()] += 1
            if tail.lower() != concept.lower():
                connected_entities.add(tail.lower())
                entity_frequency[tail.lower()] += 1
    
    # Remove the concept itself from connected entities
    connected_entities.discard(concept.lower())
    
    # Second hop: collect all possible triples first
    second_hop_triples = []
    for triple in kg_from_llm:
        head, relation, tail = triple
        # Clean up quotes
        head = head[1:-1] if "'" == head[0] or "'" == head[-1] else head
        relation = relation[1:-1] if "'" == relation[0] or "'" == relation[-1] else relation
        tail = tail[1:-1] if "'" == tail[0] or "'" == tail[-1] else tail
        
        # If any connected entity is in the triple, add it to candidates
        if head.lower() in connected_entities or tail.lower() in connected_entities:
            # Calculate relevance score based on entity frequency
            score = 0
            if head.lower() in connected_entities:
                score += entity_frequency[head.lower()]
            if tail.lower() in connected_entities:
                score += entity_frequency[tail.lower()]
            second_hop_triples.append((score, f"{head}\t{relation}\t{tail}\n"))
    
    # Sort by relevance score and take top 3
    second_hop_triples.sort(reverse=True)
    for _, triple_str in second_hop_triples[:3]:
        concept_triples += triple_str
        cnt += 1
            
    # kg_from_pubmed.json
    kg_from_pubmed = json.load(open(os.path.join(extracted_kg_path, "kg_from_pubmed.json"), "r"))
    if concept in kg_from_pubmed.keys():
        for triple in kg_from_pubmed[concept]:
            head, relation, tail = triple
            if "'" == head[0] or "'" == head[-1]:
                head = head[1:-1]
            if "'" == relation[0] or "'" == relation[-1]:
                relation = relation[1:-1]
            if "'" == tail[0] or "'" == tail[-1]:
                tail = tail[1:-1]
            concept_triples += f"{head}\t{relation}\t{tail}\n"
            cnt += 1
            
    print(f"Constructed {concept} knowledge graph with {cnt} triples")
    
    return concept_triples
    
    
def combine_concept_specific_knowledge_graphs(graph_path: str, extracted_kg_path: str):
    condition_dict, procedure_dict, drug_dict, condition_dict_inv, procedure_dict_inv, drug_dict_inv = load_mappings()

    # condition
    print("Constructing concept-specific knowledge graphs for conditions...")
    for code, name in tqdm(condition_dict.items()):
        concept_triples = read_from_extracted_kg(name, extracted_kg_path)
        with open(os.path.join(graph_path, "condition", f"{code}.txt"), "w") as f:
            f.write(concept_triples)
            
    # procedure
    print("Constructing concept-specific knowledge graphs for procedures...")
    for code, name in tqdm(procedure_dict.items()):
        concept_triples = read_from_extracted_kg(name, extracted_kg_path)
        with open(os.path.join(graph_path, "procedure", f"{code}.txt"), "w") as f:
            f.write(concept_triples)
            
    # drug
    print("Constructing concept-specific knowledge graphs for drugs...")
    for code, name in tqdm(drug_dict.items()):
        concept_triples = read_from_extracted_kg(name, extracted_kg_path)
        with open(os.path.join(graph_path, "drug", f"{code}.txt"), "w") as f:
            f.write(concept_triples)
                    
        

def main():
    graph_path = "./graphs"
    extracted_kg_path = "/shared/eng/pj20/kelpie_exp_data/kg_construct_/"
    
    if not os.path.exists(os.path.join(graph_path, "condition")) or not os.path.exists(os.path.join(graph_path, "procedure")) or not os.path.exists(os.path.join(graph_path, "drug")):
        os.makedirs(os.path.join(graph_path, "condition"))
        os.makedirs(os.path.join(graph_path, "procedure"))
        os.makedirs(os.path.join(graph_path, "drug"))
    
    combine_knowledge_graphs(graph_path, extracted_kg_path)
    combine_concept_specific_knowledge_graphs(graph_path,extracted_kg_path)

if __name__ == "__main__":
    main()