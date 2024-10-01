import os
import json
import pickle
import networkx as nx
from graspologic.partition import hierarchical_leiden
from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
import logging
import re
import time
import botocore


LLM = "Claude"  # or "Claude"
MAX_TRIPLES_PER_SUMMARY = 20
SAVE_INTERVAL = 100
MAX_COMMUNITY_SIZE = 20

TIMEOUT_SECONDS = 10  
MAX_RETRIES = 5 
MAX_SUMMARY_PER_REQUEST = 3

def read_kg(file_path: str):
    print("Reading knowledge graph...")
    g = nx.Graph()
    edge_relationships = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc="Reading triples"):
            if len(row) >= 3:
                entity1, relationship, entity2 = row[:3]

                g.add_edge(entity1, entity2)
                edge = (entity1, entity2)
                edge_relationships[edge] = relationship
                edge_relationships[(entity2, entity1)] = relationship  # Add reverse direction

    print(f"Created graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    return g, edge_relationships

def generate_community_summary(triples: list) -> str:
    # Format the triples as a bullet list
    formatted_triples = "\n".join([f"- {triple}" for triple in triples])

    prompt = \
f"""
You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical concepts and relationships provided below.

Given a list of medical triples in the format (entity1, relationship, entity2), please create a single, coherent summary that captures the key medical knowledge represented by these concepts and their relationships. The summary should be written in the third person and include all the entity names for full context.

Focus on how these medical concepts are interconnected and their relevance to medical understanding or patient care. If there are any contradictions in the triples, please resolve them in the summary.

Please provide only the summary without any starting words or phrases, and without mentioning the individual triples.

The summary should be an integrated representation of the medical knowledge contained in the triples.

Example:
Triples:
- (Diabetes, is a risk factor for, Cardiovascular Disease)
- (Hypertension, is associated with, Diabetes)
- (Obesity, contributes to, Diabetes)

Summary:
Diabetes, hypertension, and obesity are closely interconnected medical conditions. Diabetes is a significant risk factor for developing cardiovascular disease. Hypertension, or high blood pressure, is often associated with diabetes. Obesity contributes to the development of diabetes, as excess body weight can lead to insulin resistance. Managing these conditions together is crucial for reducing the risk of serious complications and improving overall patient care.

Triples:
{formatted_triples}

Summary:
"""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            if LLM == "GPT":
                response = get_gpt_response(prompt=prompt)
            elif LLM == "Claude":
                response = get_claude_response(llm="sonnet", prompt=prompt)
            else:
                raise ValueError(f"Unknown LLM: {LLM}")

            return response
        except botocore.exceptions.ReadTimeoutError:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(TIMEOUT_SECONDS)

    # If all retries failed, switch to the other LLM
    print("Switching to the other LLM...")
    if LLM == "GPT":
        response = get_claude_response(llm="sonnet", prompt=prompt)
    elif LLM == "Claude":
        response = get_gpt_response(prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")
    return response


def get_relevance(prompt: str) -> str:
    retries = 0
    while retries < MAX_RETRIES:
        try:
            if LLM == "GPT":
                response = get_gpt_response(prompt=prompt)
            elif LLM == "Claude":
                response = get_claude_response(llm="sonnet", prompt=prompt)
            else:
                raise ValueError(f"Unknown LLM: {LLM}")

            return response
        except botocore.exceptions.ReadTimeoutError:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(TIMEOUT_SECONDS)

    # If all retries failed, switch to the other LLM
    print("Switching to the other LLM...")
    if LLM == "GPT":
        response = get_claude_response(llm="sonnet", prompt=prompt)
    elif LLM == "Claude":
        response = get_gpt_response(prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")

    return response


def generate_theme_specific_summary(triples: list, theme: str) -> str:
    formatted_triples = "\n".join([f"- {triple}" for triple in triples])

    relevance_prompt = \
f"""
Given the following list of medical triples and a specific theme, please determine if the knowledge represented by these triples is relevant to the theme. Provide a simple "Yes" or "No" answer.

Triples:
{formatted_triples}

Theme: {theme}

Relevance:
"""

    relevance = get_relevance(relevance_prompt)

    if relevance.lower() == "no":
        return f"The given triples are not directly relevant to the theme: {theme}"

    prompt = \
f"""
You are a knowledgeable medical assistant tasked with generating a theme-specific summary of the medical concepts and relationships provided below.

Given a list of medical triples in the format (entity1, relationship, entity2) and a specific theme, please create a summary that focuses on how the knowledge represented by these triples is relevant to the given theme. The summary should highlight the key concepts, relationships, and implications that are most pertinent to the theme.

The summary should be written in the third person and include all the relevant entity names for context. Please provide only the summary without any starting words or phrases, and without mentioning the individual triples.

Examples:

Mortality Prediction:
Triples:
- (Diabetes, is a risk factor for, Cardiovascular Disease)
- (Hypertension, is associated with, Diabetes)
- (Obesity, contributes to, Diabetes)

Theme: Mortality prediction

Summary:
Diabetes, hypertension, and obesity are significant risk factors that can increase the likelihood of mortality. Diabetes is directly associated with an increased risk of cardiovascular disease, which is a leading cause of death. Hypertension and obesity, which often co-occur with diabetes, further compound these risks. Patients with these conditions require close monitoring and aggressive management to mitigate the risk of mortality.

Readmission Prediction:
Triples:
- (Heart Failure, is characterized by, Reduced Ejection Fraction)
- (Heart Failure, is managed with, ACE Inhibitors)
- (Heart Failure, is monitored by, B-type Natriuretic Peptide (BNP))

Theme: Readmission prediction

Summary:
Heart failure patients with reduced ejection fraction are at higher risk for hospital readmission. ACE inhibitors are a mainstay of heart failure management, helping to improve cardiac function and reduce the risk of readmission. Monitoring BNP levels can help identify patients at increased risk of readmission, as elevated BNP is associated with worsening heart failure. Close follow-up and medication adherence are critical to reducing readmission risk in these patients.

Triples:
{formatted_triples}

Theme: {theme}

Summary:
"""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            if LLM == "GPT":
                response = get_gpt_response(prompt=prompt)
            elif LLM == "Claude":
                response = get_claude_response(llm="sonnet", prompt=prompt)
            else:
                raise ValueError(f"Unknown LLM: {LLM}")

            return response
        except botocore.exceptions.ReadTimeoutError:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(TIMEOUT_SECONDS)

    # If all retries failed, switch to the other LLM
    print("Switching to the other LLM...")
    if LLM == "GPT":
        response = get_claude_response(llm="sonnet", prompt=prompt)
    elif LLM == "Claude":
        response = get_gpt_response(prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")

    return response



def combine_summaries(summaries: list) -> str:
    prompt = \
f"""
You are a knowledgeable medical assistant tasked with generating a comprehensive summary of the medical concepts and relationships based on the individual summaries provided below.

Given the following summaries of a community, please create a single, coherent summary that captures the key medical knowledge represented by this community. The combined summary should include all the important entities and relationships mentioned in the individual summaries.

Focus on the main concepts, their relationships, and their relevance to medical understanding or patient care. The summary should be concise yet informative, providing a comprehensive overview of the community's medical knowledge.

Please ensure that the combined summary is written in the third person and includes all the relevant entity names for full context.

Please provide only the summary without any starting words or phrases.

Summaries:
{summaries}

Combined Summary:
"""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            if LLM == "GPT":
                response = get_gpt_response(prompt=prompt)
            elif LLM == "Claude":
                response = get_claude_response(llm="opus", prompt=prompt)
            else:
                raise ValueError(f"Unknown LLM: {LLM}")

            return response
        except botocore.exceptions.ReadTimeoutError:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(TIMEOUT_SECONDS)
            
    # If all retries failed, switch to the other LLM
    print("Switching to the other LLM...")
    if LLM == "GPT":
        response = get_claude_response(llm="opus", prompt=prompt)
    elif LLM == "Claude":
        response = get_gpt_response(prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")

    return response


def process_community(args):
    triples, i, level = args
    if len(triples) > 150:
        seed_summaries = []
        intermediate_summaries = {}
        summary = "The community is too large"
        
    else:
        if len(triples) <= MAX_TRIPLES_PER_SUMMARY:
            summary = generate_community_summary(triples)
            seed_summaries = [summary]
            intermediate_summaries = {}
            logging.info(f"Level: {level}, Community ID: {i}\nSummary: {summary}\n")
        else:
            seed_summaries = [
                generate_community_summary(triples[j:j+MAX_TRIPLES_PER_SUMMARY])
                for j in range(0, len(triples), MAX_TRIPLES_PER_SUMMARY)
            ]
            for k, summary in enumerate(seed_summaries):
                logging.info(f"Level: {level}, Community ID: {i}, Part: {k}\nSummary: {summary}\n")

            intermediate_summaries = {}
            intermediate_level = 1

            summaries = seed_summaries
            while len(summaries) > MAX_SUMMARY_PER_REQUEST:
                new_summaries = []
                for j in range(0, len(summaries), MAX_SUMMARY_PER_REQUEST):
                    chunk_summaries = summaries[j:j+MAX_SUMMARY_PER_REQUEST]
                    new_summary = combine_summaries(chunk_summaries)
                    new_summaries.append(new_summary)
                
                intermediate_summaries[f"level {intermediate_level}"] = new_summaries
                logging.info(f"Level: {level}, Community ID: {i}, Intermediate Level: {intermediate_level}\nIntermediate Summaries: {new_summaries}\n")
                
                summaries = new_summaries
                intermediate_level += 1

            summary = combine_summaries(summaries)
            logging.info(f"Level: {level}, Community ID: {i}\nCombined Summary: {summary}\n")

    community_data = {
        "level": level,
        "community_id": i,
        "triples": triples,
        "summaries": seed_summaries,
        "intermediate_summaries": intermediate_summaries,
        "summary": summary
    }

    return community_data


def save_communities(communities, level, output_dir):
    output_file = os.path.join(output_dir, f"communities_level{level}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(communities, f, indent=4)


def read_log_file(log_file: str, level: int) -> dict:
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            log_content = file.read()

        
        summary_pattern = r"Level: (\d+), Community ID: (\d+)(, Part: (\d+))?\nSummary: (.*?)\n"
        combined_summary_pattern = r"Level: (\d+), Community ID: (\d+)\nCombined Summary: (.*?)\n"
        intermediate_summary_pattern = r"Level: (\d+), Community ID: (\d+), Intermediate Level: (\d+)\nIntermediate Summaries: (.*?)\n"

        summaries = {}

        for match in re.finditer(summary_pattern, log_content, re.DOTALL):
            level = int(match.group(1))
            community_id = int(match.group(2))
            part = int(match.group(4)) if match.group(4) else None
            summary = match.group(5)

            if level not in summaries:
                summaries[level] = {}
            if community_id not in summaries[level]:
                summaries[level][community_id] = {"summaries": [], "summary": None, "intermediate_summaries": {}}

            if part is not None:
                summaries[level][community_id]["summaries"].append(summary)
            else:
                summaries[level][community_id]["summaries"] = [summary]
                summaries[level][community_id]["summary"] = summary
            

        for match in re.finditer(combined_summary_pattern, log_content, re.DOTALL):
            level = int(match.group(1))
            community_id = int(match.group(2))
            combined_summary = match.group(3)

            if level in summaries and community_id in summaries[level]:
                summaries[level][community_id]["summary"] = combined_summary
                
        for match in re.finditer(intermediate_summary_pattern, log_content, re.DOTALL):
            level = int(match.group(1))
            community_id = int(match.group(2))
            intermediate_level = int(match.group(3))
            intermediate_summaries = eval(match.group(4))

            if level not in summaries:
                summaries[level] = {}
            if community_id not in summaries[level]:
                summaries[level][community_id] = {"summaries": [], "summary": None, "intermediate_summaries": {}}

            summaries[level][community_id]["intermediate_summaries"][f"level {intermediate_level}"] = intermediate_summaries
    else:
        summaries = {level: {}}
            
    return summaries

def set_log_file_for_level(level, output_dir):
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers

    # Create a file handler for the new log file
    log_file = os.path.join(output_dir, f"community_summaries_{level}.log")
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    handler.setFormatter(formatter)

    # Add the new handler to the logger
    logger.addHandler(handler)

def partition_and_summarize(g: nx.Graph, edge_relationships: dict, max_cluster_size: int, output_dir: str, log_file: str, read_log=False) -> None:    
    print(f"Partitioning graph with max_cluster_size: {max_cluster_size}")
    community_mapping = hierarchical_leiden(g, max_cluster_size=max_cluster_size, random_seed=42)

    node_to_community = {}
    for partition in community_mapping:
        level = partition.level
        node = partition.node
        community = partition.cluster

        if level not in node_to_community:
            node_to_community[level] = {}
        node_to_community[level][node] = community

    # print out the community size for each level
    for level, node_to_community_map in node_to_community.items():
        print(f"Level {level}: {len(set(node_to_community_map.values()))} communities")

    for level, node_to_community_map in node_to_community.items():
        # if level == 3 or level == 0 or level == 1 or level == 2:
        #     continue
        current_log_file = log_file.replace(".log", f"_{level}.log")
        
        set_log_file_for_level(level, output_dir)
        
        if read_log:
            summaries = read_log_file(current_log_file, level=level)
        print(f"Processing level {level} with {len(set(node_to_community_map.values()))} communities")

        communities = {}
        unprocessed_communities = []

        for community_id in tqdm(set(node_to_community_map.values())):
            community_nodes = [n for n in g.nodes if node_to_community_map.get(n) == community_id]
            triples = [
                (u, edge_relationships[(u, v)], v)
                for u, v in g.edges(community_nodes)
                if u in community_nodes and v in community_nodes
            ]
            if read_log:
                if community_id in summaries[level]:
                    community_summaries = summaries[level][community_id]["summaries"]
                    community_summary = summaries[level][community_id]["summary"]
                    intermediate_summaries = summaries[level][community_id]["intermediate_summaries"]

                    if community_summary is None:
                        unprocessed_communities.append((triples, community_id, level))
                        continue
                        
                        
                        # print(f"Resuming summarization for community {community_id} at level {level}")
                        
                        # if len(community_summaries) < len(triples) // MAX_TRIPLES_PER_SUMMARY:
                        #     # Resume generating individual summaries
                        #     start_index = len(community_summaries) * MAX_TRIPLES_PER_SUMMARY
                        #     remaining_triples = triples[start_index:]
                            
                        #     for j in range(0, len(remaining_triples), MAX_TRIPLES_PER_SUMMARY):
                        #         sub_triples = remaining_triples[j:j+MAX_TRIPLES_PER_SUMMARY]
                        #         sub_summary = generate_community_summary(sub_triples)
                        #         community_summaries.append(sub_summary)
                        #         logging.info(f"Level: {level}, Community ID: {community_id}, Part: {len(community_summaries)-1}\nSummary: {sub_summary}\n")

                        # last_summaries = community_summaries

                        # if intermediate_summaries:
                        #     last_intermediate_level = max(intermediate_summaries.keys())
                        #     last_summaries = intermediate_summaries[last_intermediate_level]
                        # else:
                        #     last_intermediate_level = "level 0"

                        # while len(last_summaries) > MAX_SUMMARY_PER_REQUEST:
                        #     new_summaries = []
                        #     for j in range(0, len(last_summaries), MAX_SUMMARY_PER_REQUEST):
                        #         chunk_summaries = last_summaries[j:j+MAX_SUMMARY_PER_REQUEST]
                        #         new_summary = combine_summaries(chunk_summaries)
                        #         new_summaries.append(new_summary)

                        #     intermediate_level = int(last_intermediate_level.split()[-1]) + 1
                        #     intermediate_summaries[f"level {intermediate_level}"] = new_summaries
                        #     logging.info(f"Level: {level}, Community ID: {community_id}, Intermediate Level: {intermediate_level}\nIntermediate Summaries: {new_summaries}\n")

                        #     last_summaries = new_summaries
                        #     last_intermediate_level = f"level {intermediate_level}"

                        # community_summary = combine_summaries(last_summaries)
                        # logging.info(f"Level: {level}, Community ID: {community_id}\nCombined Summary: {community_summary}\n")
                        

                    community_data = {
                        "level": level,
                        "community_id": community_id,
                        "triples": triples,
                        "summaries": community_summaries,
                        "intermediate_summaries": intermediate_summaries,
                        "summary": community_summary
                    }

                    communities[community_id] = community_data
                else:
                    unprocessed_communities.append((triples, community_id, level))
            else:
                
                unprocessed_communities.append((triples, community_id, level))

        save_communities(communities, level, output_dir)
        
        print(f"Unprocessed communities at level {level}: {len(unprocessed_communities)}")
        # Process unprocessed communities using multi-threading
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_community, item) for item in unprocessed_communities]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing unprocessed communities (level={level})"):
                community_data = future.result()
                community_id = community_data["community_id"]
                communities[community_id] = community_data

        # Save updated communities to JSON file
        save_communities(communities, level, output_dir)

        print(f"Completed processing for level {level}")

    print("Summaries extracted and saved successfully.")
        

def main():
    kg_path = "../graph/kg_refined.txt"
    output_dir = "/shared/eng/pj20/kelpie_exp_data/indexing"
    log_file = f"{output_dir}/community_summaries.log"
    os.makedirs(output_dir, exist_ok=True)
    # Add this line to configure the logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    if os.path.exists(os.path.join(output_dir, "edge_relationships.pkl")) and os.path.exists(os.path.join(output_dir, "kg.gpickle")):
        print("Combined knowledge graph and edge relationships already exist, reading...")
        g = nx.read_gpickle(os.path.join(output_dir, "kg.gpickle"))
        with open(os.path.join(output_dir, "edge_relationships.pkl"), 'rb') as f:
            edge_relationships = pickle.load(f)
    else:
        print(f"Reading combined knowledge graph from: {kg_path}")
        g, edge_relationships = read_kg(kg_path)
        print(f"Read graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        nx.write_gpickle(g, os.path.join(output_dir, "kg.gpickle"))
        with open(os.path.join(output_dir, "edge_relationships.pkl"), 'wb') as f:
            pickle.dump(edge_relationships, f)

    partition_and_summarize(g, edge_relationships, MAX_COMMUNITY_SIZE, output_dir, log_file, read_log=False)

if __name__ == "__main__":
    main()