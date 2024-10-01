import os
import json
import networkx as nx
from graspologic.partition import hierarchical_leiden
from tqdm import tqdm
import csv
import logging
import random

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

def partition_graph(g: nx.Graph, edge_relationships: dict, max_cluster_size: int, output_dir: str) -> None:
    print(f"Partitioning graph with max_cluster_size: {max_cluster_size}")

    unique_communities = set()
    iteration = 0
    output_data = []

    while len(unique_communities) < 60000:
        print(f"Running iteration {iteration}...")
        print(f"Unique communities: {len(unique_communities)}")
        randomness = random.uniform(0.01, 1.0)
        community_mapping = hierarchical_leiden(g, max_cluster_size=max_cluster_size, random_seed=int(iteration+1), randomness=randomness)

        node_to_community = {}
        for partition in community_mapping:
            level = partition.level
            node = partition.node
            community = partition.cluster
            
            if level not in node_to_community:
                node_to_community[level] = {}
            node_to_community[level][node] = community
                
        for level, node_to_community_map in node_to_community.items():
            for community_id in tqdm(set(node_to_community_map.values())):
                community_nodes = [n for n in g.nodes if node_to_community_map.get(n) == community_id]
                triples = [
                    tuple(sorted([u, edge_relationships[(u, v)], v]))
                    for u, v in g.edges(community_nodes)
                    if u in community_nodes and v in community_nodes
                ]

                triples_set = frozenset(triples)
                if triples_set not in unique_communities:
                    unique_communities.add(triples_set)
                    community_data = {
                        "run": iteration,
                        "level": level,
                        "community_id": community_id,
                        "triples": list(triples),
                        "randomness": randomness
                    }
                    output_data.append(community_data)

                    if len(unique_communities) >= 60000:
                        break

        iteration += 1

    output_file = os.path.join(output_dir, "communities.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print("Graph partitioning and community data output completed.")

def main():
    kg_path = "../graph/kg_refined.txt"
    output_dir = "/shared/eng/pj20/kelpie_exp_data/indexing_1"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading combined knowledge graph from: {kg_path}")
    g, edge_relationships = read_kg(kg_path)
    print(f"Read graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")

    MAX_COMMUNITY_SIZE = 10
    partition_graph(g, edge_relationships, MAX_COMMUNITY_SIZE, output_dir)

if __name__ == "__main__":
    main()