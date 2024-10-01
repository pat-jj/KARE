import csv
from tqdm import tqdm
import os
import json
import random
from langdetect import detect
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
import sys


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

def find_shortest_paths(graph, start, end, max_length=5, max_paths=10, max_nodes=3000):
    paths = []
    queue = [(start, [start])]
    visited = set()
    nodes_explored = 0
    while queue and len(paths) < max_paths and nodes_explored < max_nodes:
        (node, path) = queue.pop(0)
        nodes_explored += 1
        if node == end:
            paths.append(path)
            continue
        if len(path) > max_length:
            continue
        if node not in visited:
            visited.add(node)
            for relation, neighbor in graph.get(node, []):
                if neighbor not in path:
                    queue.append((neighbor, path + [relation, neighbor]))
    return paths if paths else None


def get_term(cui, concept, concept_name, umls_mapping, related_concepts_names):
    if cui in related_concepts_names:
        return related_concepts_names[cui]
    elif cui == concept:
        return concept_name
    else:
        terms = [term for term in umls_mapping[cui] if is_english(term)]
        return terms[0] if terms else None
    
    
def find_shortest_paths_bidirectional(graph, start, end, max_length=6, max_paths=20, max_nodes=2000):
    from collections import deque

    if start == end:
        return [[start]]

    # Forward and backward search queues and visited dictionaries
    queue_forward = deque([[start]])
    queue_backward = deque([[end]])
    visited_forward = {start: [start]}
    visited_backward = {end: [end]}

    paths = []
    nodes_explored = 0

    while queue_forward and queue_backward and len(paths) < max_paths and nodes_explored < max_nodes:
        # Expand forward
        path_forward = queue_forward.popleft()
        last_node_forward = path_forward[-1]

        if last_node_forward in visited_backward:
            # Path found
            full_path = path_forward + visited_backward[last_node_forward][::-1][1:]
            paths.append(full_path)
            if len(paths) >= max_paths:
                break

        if len(path_forward) // 2 <= max_length:
            for relation, neighbor in graph.get(last_node_forward, []):
                if neighbor not in visited_forward:
                    visited_forward[neighbor] = path_forward + [relation, neighbor]
                    queue_forward.append(visited_forward[neighbor])
                    nodes_explored += 1

        # Expand backward
        path_backward = queue_backward.popleft()
        last_node_backward = path_backward[-1]

        if last_node_backward in visited_forward:
            # Path found
            full_path = visited_forward[last_node_backward] + path_backward[::-1][1:]
            paths.append(full_path)
            if len(paths) >= max_paths:
                break

        if len(path_backward) // 2 <= max_length:
            for relation, neighbor in graph.get(last_node_backward, []):
                if neighbor not in visited_backward:
                    visited_backward[neighbor] = path_backward + [relation, neighbor]
                    queue_backward.append(visited_backward[neighbor])
                    nodes_explored += 1

    return paths if paths else None

def process_concept_parallel(args):
    # Unpack arguments
    concept, related_concepts, concept_name, related_concepts_names, umls_graph, umls_mapping, path_cache, task_id = args

    local_output = {}
    for related_concept in related_concepts:
        if concept in umls_mapping and related_concept in umls_mapping:
            start_cui = concept
            end_cui = related_concept

            # Check if paths are in cache
            if (start_cui, end_cui) in path_cache:
                paths = path_cache[(start_cui, end_cui)]
            else:
                paths = find_shortest_paths_bidirectional(umls_graph, start_cui, end_cui, max_length=7, max_paths=40, max_nodes=10000)
                path_cache[(start_cui, end_cui)] = paths

            if paths:
                triples = set()
                path_with_terms = []
                for path in paths:
                    for i in range(0, len(path) - 2, 2):
                        head = path[i]
                        relation = path[i + 1]
                        tail = path[i + 2]
                        if head in umls_mapping and tail in umls_mapping:
                            head_term = get_term(head, concept, concept_name, umls_mapping, related_concepts_names)
                            tail_term = get_term(tail, concept, concept_name, umls_mapping, related_concepts_names)
                            if head_term and tail_term:
                                try:
                                    triples.add((head_term, relation, tail_term))
                                except Exception as e:
                                    print(f"Error adding triple ({head_term}, {relation}, {tail_term}): {e}")
                    # Also, construct path with terms
                    path_terms = []
                    for i in range(len(path)):
                        if i % 2 == 0:
                            cui = path[i]
                            term = get_term(cui, concept, concept_name, umls_mapping, related_concepts_names)
                            path_terms.append(term if term else cui)
                        else:
                            relation = path[i]
                            path_terms.append(relation)
                    path_with_terms.append(path_terms)

                # Store in local_output
                if concept_name not in local_output:
                    local_output[concept_name] = {}
                related_concept_name = related_concepts_names.get(related_concept, '')
                if related_concept_name not in local_output[concept_name]:
                    local_output[concept_name][related_concept_name] = {'triples': [], 'paths': []}
                local_output[concept_name][related_concept_name]['triples'].extend(list(triples))
                local_output[concept_name][related_concept_name]['paths'].extend(path_with_terms)

                # Remove duplicates
                local_output[concept_name][related_concept_name]['triples'] = list(set(local_output[concept_name][related_concept_name]['triples']))
                local_output[concept_name][related_concept_name]['paths'] = [list(x) for x in set(tuple(x) for x in local_output[concept_name][related_concept_name]['paths'])]

    return task_id, local_output


def process_concept(concept, related_concepts, concept_name, related_concepts_names, umls_graph, umls_mapping, output_file, path_cache):
    for related_concept in related_concepts:
        if concept in umls_mapping and related_concept in umls_mapping:
            start_cui = concept
            end_cui = related_concept

            # Check if paths are in cache
            if (start_cui, end_cui) in path_cache:
                paths = path_cache[(start_cui, end_cui)]
            else:
                paths = find_shortest_paths(umls_graph, start_cui, end_cui, max_length=3, max_paths=10)
                path_cache[(start_cui, end_cui)] = paths

            if paths:
                triples = set()
                path_with_terms = []
                for path in paths:
                    for i in range(0, len(path) - 2, 2):
                        head = path[i]
                        relation = path[i + 1]
                        tail = path[i + 2]
                        if head in umls_mapping and tail in umls_mapping:
                            head_term = get_term(head, concept, concept_name, umls_mapping, related_concepts_names)
                            tail_term = get_term(tail, concept, concept_name, umls_mapping, related_concepts_names)
                            if head_term and tail_term:
                                try:
                                    triples.add((head_term, relation, tail_term))
                                except Exception as e:
                                    print(f"Error adding triple ({head_term}, {relation}, {tail_term}): {e}")
                    # Also, construct path with terms
                    path_terms = []
                    for i in range(len(path)):
                        if i % 2 == 0:
                            cui = path[i]
                            term = get_term(cui, concept, concept_name, umls_mapping, related_concepts_names)
                            path_terms.append(term if term else cui)
                        else:
                            relation = path[i]
                            path_terms.append(relation)
                    path_with_terms.append(path_terms)

                # Store in output_file
                if concept_name not in output_file:
                    output_file[concept_name] = {}
                related_concept_name = related_concepts_names.get(related_concept, '')
                if related_concept_name not in output_file[concept_name]:
                    output_file[concept_name][related_concept_name] = {'triples': [], 'paths': []}
                output_file[concept_name][related_concept_name]['triples'].extend(list(triples))
                output_file[concept_name][related_concept_name]['paths'].extend(path_with_terms)

                # Remove duplicates
                output_file[concept_name][related_concept_name]['triples'] = list(set(output_file[concept_name][related_concept_name]['triples']))
                output_file[concept_name][related_concept_name]['paths'] = [list(x) for x in set(tuple(x) for x in output_file[concept_name][related_concept_name]['paths'])]

def main():
    resource_path = "./resources"
    condition_dict, procedure_dict, drug_dict = read_code2name(resource_path)
    condition_dict_inv = {v: k for k, v in condition_dict.items()}
    procedure_dict_inv = {v: k for k, v in procedure_dict.items()}
    drug_dict_inv = {v: k for k, v in drug_dict.items()}
    
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
    ccscm_to_umls = {ccscm: icd9_to_umls[icd9] for ccscm, icd9 in ccscm_to_icd9cm.items() if icd9 in icd9_to_umls}
    ccsproc_to_umls = {ccsproc: icd9_to_umls[icd9] for ccsproc, icd9 in ccsproc_to_icd9proc.items() if icd9 in icd9_to_umls}
    print("Mappings loaded")

    graph_path = "./graphs"
    os.makedirs(graph_path, exist_ok=True)

    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/all_top_coexisting_concepts.json", "r") as f:
        all_top_coexisting_concepts = json.load(f)

    # Extract subgraphs for medical concepts
    print("Extracting subgraphs for medical concepts...")
    output_file = {}
    manager = defaultdict(dict)  # Shared dictionary across processes
    path_cache = {}  # Global cache for storing paths between concepts

    # Prepare arguments for multiprocessing
    tasks = []
    for concept, related_concepts in tqdm(all_top_coexisting_concepts.items()):
        if concept in condition_dict_inv:
            concept_name = concept
            related_concepts_names = {}
            concept = condition_dict_inv[concept]
            if concept in ccscm_to_umls:
                concept = ccscm_to_umls[concept][0]
            else:
                continue
            related_concepts_ = []
            for related_concept in related_concepts:
                if related_concept in condition_dict_inv:
                    related_concept_name = related_concept
                    related_concept = condition_dict_inv[related_concept]
                    related_concept = ccscm_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in procedure_dict_inv:
                    related_concept_name = related_concept
                    related_concept = procedure_dict_inv[related_concept]
                    related_concept = ccsproc_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in drug_dict_inv:
                    related_concept_name = related_concept
                    related_concept = drug_dict_inv[related_concept]
                    related_concept = atc_to_umls[related_concept][0] if type(atc_to_umls[related_concept]) == list else atc_to_umls[related_concept]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name

            tasks.append((concept, related_concepts_, concept_name, related_concepts_names, umls_graph, umls_mapping, path_cache))
            
        elif concept in procedure_dict_inv:
            concept_name = concept
            related_concepts_names = {}
            concept = procedure_dict_inv[concept]
            if concept in ccsproc_to_umls:
                concept = ccsproc_to_umls[concept][0]
            else:
                continue
            related_concepts_ = []
            for related_concept in related_concepts:
                if related_concept in condition_dict_inv:
                    related_concept_name = related_concept
                    related_concept = condition_dict_inv[related_concept]
                    related_concept = ccscm_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in procedure_dict_inv:
                    related_concept_name = related_concept
                    related_concept = procedure_dict_inv[related_concept]
                    related_concept = ccsproc_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in drug_dict_inv:
                    related_concept_name = related_concept
                    related_concept = drug_dict_inv[related_concept]
                    related_concept = atc_to_umls[related_concept][0] if type(atc_to_umls[related_concept]) == list else atc_to_umls[related_concept]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name

            tasks.append((concept, related_concepts_, concept_name, related_concepts_names, umls_graph, umls_mapping, path_cache))
            
        elif concept in drug_dict_inv:
            concept_name = concept
            related_concepts_names = {}
            concept = drug_dict_inv[concept]
            if concept in atc_to_umls:
                concept = atc_to_umls[concept][0] if type(atc_to_umls[concept]) == list else atc_to_umls[concept]
            else:
                continue
            related_concepts_ = []
            for related_concept in related_concepts:
                if related_concept in condition_dict_inv:
                    related_concept_name = related_concept
                    related_concept = condition_dict_inv[related_concept]
                    related_concept = ccscm_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in procedure_dict_inv:
                    related_concept_name = related_concept
                    related_concept = procedure_dict_inv[related_concept]
                    related_concept = ccsproc_to_umls[related_concept][0]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name
                elif related_concept in drug_dict_inv:
                    related_concept_name = related_concept
                    related_concept = drug_dict_inv[related_concept]
                    related_concept = atc_to_umls[related_concept][0] if type(atc_to_umls[related_concept]) == list else atc_to_umls[related_concept]
                    related_concepts_.append(related_concept)
                    related_concepts_names[related_concept] = related_concept_name

            tasks.append((concept, related_concepts_, concept_name, related_concepts_names, umls_graph, umls_mapping, path_cache))

    # Use multiprocessing Pool
    # num_workers = max(cpu_count() - 1, 1)
    
    completed_tasks = 0
    total_tasks = len(tasks)
    tasks_with_id = [(task[0], task[1], task[2], task[3], task[4], task[5], task[6], i) for i, task in enumerate(tasks)]
    
    with Pool(processes=15) as pool:
        for task_id, local_output in pool.imap_unordered(process_concept_parallel, tasks_with_id):
            completed_tasks += 1
            print(f"Completed task {completed_tasks}/{total_tasks}")

            for concept_name, related_data in local_output.items():
                if concept_name not in output_file:
                    output_file[concept_name] = {}
                for related_concept_name, data in related_data.items():
                    if related_concept_name not in output_file[concept_name]:
                        output_file[concept_name][related_concept_name] = {'triples': [], 'paths': []}
                    output_file[concept_name][related_concept_name]['triples'].extend(data['triples'])
                    output_file[concept_name][related_concept_name]['paths'].extend(data['paths'])
                    # Remove duplicates
                    output_file[concept_name][related_concept_name]['triples'] = list(set(output_file[concept_name][related_concept_name]['triples']))
                    output_file[concept_name][related_concept_name]['paths'] = [list(x) for x in set(tuple(x) for x in output_file[concept_name][related_concept_name]['paths'])]
            
            if completed_tasks % 100 == 0:
                print("Performing intermediate saving...")
                with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/kg_from_kg_intermediate.json", "w") as f:
                    json.dump(output_file, f, indent=2)
                    
            

    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/kg_from_kg.json", "w") as f:
        json.dump(output_file, f, indent=2)

if __name__ == "__main__":
    main()
