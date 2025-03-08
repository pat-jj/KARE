import json
from typing import List, Set, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from apis.claude_api import get_claude_response
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


SIMILARITY_THRESHOLD = 5
MAX_THREADS = 15

def filter_similar_sets(concept_sets: List[Set[str]], similarity_threshold: int = SIMILARITY_THRESHOLD) -> List[Set[str]]:
    """Filter out highly similar concept sets based on the concept multihot vector."""
    vectorizer = CountVectorizer(vocabulary=list(set(concept for concept_set in concept_sets for concept in concept_set)))
    concept_vectors = vectorizer.transform([' '.join(concept_set) for concept_set in concept_sets])
    filtered_sets = []
    processed_indices = set()

    for i, concept_set in enumerate(concept_sets):
        if i not in processed_indices:
            similar_indices = [j for j, other_set in enumerate(concept_sets) if i != j and
                               len(concept_set.symmetric_difference(other_set)) < similarity_threshold]
            filtered_sets.append(concept_set)
            processed_indices.update(similar_indices)

    return filtered_sets


def extract_relationships_llm(concepts: List[str]) -> str:
    """Extract relationships from LLM."""
    prompt = \
f"""Please identify the relationships among these medical concepts that can be potentially helpful to clinical predictions (e.g., mortality prediction, readmission prediction) as many as possible.

You can introduce intermediate relationships with other entities based on your knowledge. 

Consider how these concepts would interact with others to be useful for clinical predictions. There's no need to keep all the relationships connected.

For the concepts provided in the list, you MUST use the their original name without any changes.  Please output only the list of triples without any other information.

Output format: 
[[ENTITY1, RELATIONSHIP_1, ENTITY2], 
[ENTITY2, RELATIONSHIP_2, ENTITY3], ...]

Medical Concepts:
{concepts}

Output:
"""

    response = get_claude_response(llm="sonnet", prompt=prompt)
    return response

def parse_triples(response: str) -> List[Tuple[str, str, str]]:
    """Parse triples from LLM response."""
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, response.strip())
    
    triples = []
    for match in matches:
        triple = [part.strip() for part in match.split(',')]
        if len(triple) == 3:
            entity1, relation, entity2 = triple
            entity1 = entity1.replace("[", "").replace("]", "")
            entity2 = entity2.replace("[", "").replace("]", "")
            triples.append((entity1, relation, entity2))
    
    return triples

def process_concept_set(concept_set: Set[str]) -> dict:
    concepts = list(concept_set)
    response = extract_relationships_llm(concepts)
    triples = parse_triples(response)
    
    return triples

def process_concept_set_(concept_set: Set[str]) -> dict:
    concepts = list(concept_set)
    response = extract_relationships_llm(concepts)
    triples = parse_triples(response)
    
    # Create a graph representation
    graph = {}
    for triple in triples:
        entity1, relation, entity2 = triple
        if entity1 not in graph:
            graph[entity1] = set()
        if entity2 not in graph:
            graph[entity2] = set()
        graph[entity1].add((relation, entity2, False))  # False indicates original direction
        graph[entity2].add((f"inverse_{relation}", entity1, True))  # True indicates inverse
    
    kg_triples_dict_ = {}
    
    def get_2hop_subgraph(start_concept):
        subgraph = set()
        visited = set()
        queue = [(start_concept, 0)]
        
        while queue:
            current_concept, depth = queue.pop(0)
            if depth > 2:
                break
            
            if current_concept in graph:
                for relation, neighbor, is_inverse in graph[current_concept]:
                    if is_inverse:
                        triple = (neighbor, relation.replace("inverse_", ""), current_concept)
                    else:
                        triple = (current_concept, relation, neighbor)
                    subgraph.add(triple)
                    if neighbor not in visited and depth < 2:
                        queue.append((neighbor, depth + 1))
                        visited.add(neighbor)
        
        return subgraph
    
    for concept in concept_set:
        kg_triples_dict_[concept] = list(get_2hop_subgraph(concept))
    
    return kg_triples_dict_

def build_kg_for_concepts_llm(concept_sets: List[Set[str]], save_interval: int = 1000) -> dict:
    kg_triples_set = set()
    processed_count = 0
    skipped_count = 0
    
    def process_concept_set_wrapper(concept_set: Set[str]) -> Tuple[dict, bool]:
        try:
            result = process_concept_set(concept_set)
            return result, True
        except Exception as e:
            print(f"Error processing concept set: {e}")
            return {}, False
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_concept_set_wrapper, concept_set) for concept_set in concept_sets]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, success = future.result()
            if success:
                kg_triples_set.update(set(result))
                processed_count += 1
            else:
                skipped_count += 1
            
            if (processed_count + skipped_count) % save_interval == 0:
                print(f"Processed: {processed_count}, Skipped: {skipped_count}")
                with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/graphs/llm/kg_from_llm_{processed_count}.json", "w") as f:
                    json.dump(list(kg_triples_set), f, indent=4)
                    
    return kg_triples_set
                
    
    # # Convert sets to lists for JSON serialization
    # for concept in kg_triples_dict:
    #     kg_triples_dict[concept] = list(kg_triples_dict[concept])
    
    # print(f"Final count - Processed: {processed_count}, Skipped: {skipped_count}")
    # return kg_triples_dict


# def save_kg_triples(kg_triples_dict: dict, file_path: str):
#     # Convert sets to lists for JSON serialization
#     serializable_dict = {concept: list(triples) for concept, triples in kg_triples_dict.items()}
#     with open(file_path, "w") as f:
#         json.dump(serializable_dict, f, indent=4)


def main():
    # Load the all_visit_concepts from the JSON file
    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/all_visit_concepts.json", "r") as f:
        all_visit_concepts = [set(visit) for visit in json.load(f)]

    if os.path.exists(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json"):
        print("Loading filtered concept sets from file...")
        with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "r") as f:
            filtered_concept_sets = json.load(f)
    else:
        print("Filtering similar concept sets...")
        filtered_concept_sets = filter_similar_sets(all_visit_concepts)

        with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "w") as f:
            json.dump(filtered_concept_sets, f)

    # Build knowledge graph for filtered concept sets using LLM
    kg_triples_dict_llm = build_kg_for_concepts_llm(filtered_concept_sets, save_interval=500)

    # Save the final KG triples from LLM to a JSON file
    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct_/kg_from_llm.json", "w") as f:
        json.dump(list(kg_triples_dict_llm), f, indent=4)

    print(f"Number of filtered concept sets: {len(filtered_concept_sets)}")  
    print(f"Number of concepts with triples from LLM: {len(kg_triples_dict_llm)}")
    
    
if __name__ == "__main__":
    main()