import os
from collections import defaultdict
from typing import List, Tuple


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


def read_triples(file_path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def combine_knowledge_graphs(graph_path: str) -> None:
    combined_triples = defaultdict(set)
    
    for mode in ["condition", "procedure", "drug"]:
        for source in ["Claude", "PubMed"]: 
            mode_path = os.path.join("theme_specific_graphs", mode, source)
            
            if os.path.exists(mode_path):
                for file_name in os.listdir(mode_path):
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(mode_path, file_name)
                        triples = read_triples(file_path)
                        
                        for triple in triples:
                            entity1, relationship, entity2 = triple
                            # Map the relationship to its full description only if it's one of the specified abbreviations
                            if relationship in REL_MAPPING:
                                relationship = REL_MAPPING[relationship]
                            combined_triples[(entity1, relationship, entity2)].add(source)
    
    output_file = os.path.join(graph_path, "kg_theme_raw.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triple, sources in combined_triples.items():
            entity1, relationship, entity2 = triple
            source_str = ",".join(sorted(sources))
            f.write(f"{entity1.lower()}\t{relationship.lower()}\t{entity2.lower()}\n")
    
    print(f"Combined knowledge graph saved to: {output_file}")
    

def combine_concept_specific_knowledge_graphs():
    for mode in ["condition", "procedure", "drug"]:
        for source in ["Claude", "PubMed"]: 
            mode_path = os.path.join("theme_specific_graphs", mode, source)
            combined_path = os.path.join("theme_specific_graphs", mode, "combined")
            if not os.path.exists(combined_path):
                os.makedirs(combined_path)
            
            for file_name in os.listdir(mode_path):
                if file_name.endswith(".txt") and "_" not in file_name:
                    file_path = os.path.join(mode_path, file_name)
                    triples = read_triples(file_path)
                    
                    output_file = os.path.join(combined_path, file_name)
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for triple in triples:
                            entity1, relationship, entity2 = triple
                            if relationship in REL_MAPPING:
                                relationship = REL_MAPPING[relationship]
                            f.write(f"{entity1.lower()}\t{relationship.lower()}\t{entity2.lower()}\n")
                            
                print(f'merged {source} {mode} {file_name} to combined')

    

def main():
    graph_path = "../graph"
    combine_knowledge_graphs(graph_path)
    combine_concept_specific_knowledge_graphs()

if __name__ == "__main__":
    main()