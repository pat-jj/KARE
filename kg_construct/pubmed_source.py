# import json
# from typing import List, Set, Tuple
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.sparse import csr_matrix
# from urllib.request import urlopen
# from urllib.parse import urlencode
# from apis.gpt_api import get_gpt_response
# import xml.etree.ElementTree as ET
# from apis.claude_api import get_claude_response
# import time
# import re
# import os
# from pubmed_index.abstract_retriever import AbstractRetriever
# from tqdm import tqdm

# SIMILARITY_THRESHOLD = 5

# db_file = "/shared/eng/pj20/kelpie_exp_data/pubmed_vectors/pubmed_data.db"
# h5_file = "/shared/eng/pj20/kelpie_exp_data/pubmed_vectors/pubmed_embeddings.h5"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# retriever = AbstractRetriever(h5_file, db_file, chunk_size=250000, use_cuda=True)
    
# def filter_similar_sets(concept_sets: List[Set[str]], similarity_threshold: int = 5) -> List[Set[str]]:
#     """Filter out highly similar concept sets based on the concept multihot vector."""
#     vectorizer = CountVectorizer(vocabulary=list(set(concept for concept_set in concept_sets for concept in concept_set)))
#     concept_vectors = vectorizer.transform([' '.join(concept_set) for concept_set in concept_sets])
#     filtered_sets = []
#     processed_indices = set()

#     for i, concept_set in enumerate(concept_sets):
#         if i not in processed_indices and len(concept_set) > 10:
#             similar_indices = [j for j, other_set in enumerate(concept_sets) if i != j and len(other_set) > 10 and
#                                len(concept_set.symmetric_difference(other_set)) < similarity_threshold]
#             filtered_sets.append(concept_set)
#             processed_indices.update(similar_indices)
#         elif i not in processed_indices:
#             filtered_sets.append(concept_set)

#     return filtered_sets


# def build_kg_for_concepts(concept_sets: List[Set[str]]) -> List[Tuple[str, str, str]]:
#     """Build knowledge graph for medical concepts using PubMed abstracts and LLM."""
#     kg_triples = []

#     for concept_set in tqdm(concept_sets):
#         terms = list(concept_set)
#         top_k = 10
#         pmids, distances, abstracts = retriever.search(" ".join(terms), top_k=top_k)

#         for abstract in abstracts:
#             response = extract_relationships(abstract, terms)
#             triples = parse_triples(response, concept_set)

#             for triple in triples:
#                 for concept in concept_set:
#                     if concept in triple:
#                         if concept not in kg_triples:
#                             kg_triples[concept] = []
#                         kg_triples[concept].append(triple)
#                         break

#     return kg_triples


# def extract_relationships(text: str, concepts: List[str]) -> str:
#     """Extract relationships from text using LLM."""
#     example = """Example:
# Text:
# Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. Common symptoms include wheezing, coughing, shortness of breath, and chest tightness. Triggers can vary but often include allergens, air pollution, exercise, and respiratory infections. Management typically involves a combination of long-term control medications, such as inhaled corticosteroids, and quick-relief medications like short-acting beta-agonists. Recent research has focused on personalized treatment approaches, including biologics for severe asthma and the role of the microbiome in asthma development and progression. Proper inhaler technique and adherence to medication regimens are crucial for effective management. Asthma action plans, developed in partnership with healthcare providers, help patients manage symptoms and exacerbations.

# Concepts: [asthma, inflammation, airways, wheezing, coughing, inhaled corticosteroids, short-acting beta-agonists, allergens, respiratory infections]

# Extracted triples:
# [[asthma, is a, chronic respiratory condition], [asthma, characterized by, inflammation of airways], [inflammation, causes, narrowing of airways], [narrowing of airways, leads to, breathing difficulties], [wheezing, is a symptom of, asthma], [coughing, is a symptom of, asthma], [allergens, can trigger, asthma], [respiratory infections, can trigger, asthma], [inhaled corticosteroids, used for, long-term control of asthma], [short-acting beta-agonists, provide, quick relief in asthma]]
# """

#     prompt = f"""Given a medical text and a list of important concepts, extract relevant relationships between the concepts from the text (if present). For each triple, if an entity matches one of the given concepts, replace the entity with the exact concept term.

# Focus on generating high-quality triples closely related to the provided concepts. Aim to extract at most 10 triples for each text. Each triple should follow this format: [ENTITY1, RELATIONSHIP, ENTITY2]. Ensure the triples are informative and logically sound.

# {example}

# Text:
# {text}

# Concepts: {concepts}

# Extracted triples:
# """

#     response = get_claude_response(llm="sonnet", prompt=prompt)
#     return response


# def parse_triples(response: str, concept_set: Set[str]) -> List[Tuple[str, str, str]]:
#     """Parse triples from LLM response."""
#     pattern = r"\[(.*?)\]"
#     matches = re.findall(pattern, response)
    
#     triples = []
#     for match in matches:
#         triple = match.split(', ')
#         if len(triple) == 3:
#             entity1, relation, entity2 = triple
#             entity1 = entity1.replace("[", "").replace("]", "")
#             entity2 = entity2.replace("[", "").replace("]", "")
#             triples.append((entity1, relation, entity2))
    
#     return triples

# def main():

#     # Load the all_visit_concepts from the JSON file
#     with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/all_visit_concepts.json", "r") as f:
#         all_visit_concepts = [set(visit) for visit in json.load(f)]

#     if os.path.exists(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json"):
#         print("Loading filtered concept sets from file...")
#         with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "r") as f:
#             filtered_concept_sets = json.load(f)
#     else:
#         print("Filtering similar concept sets...")
#         filtered_concept_sets = filter_similar_sets(all_visit_concepts)

#         with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "w") as f:
#             json.dump(filtered_concept_sets, f)

#     # Filter out highly similar concept sets
#     filtered_concept_sets = filter_similar_sets(all_visit_concepts)

#     # Build knowledge graph for filtered concept sets
#     kg_triples_dict = build_kg_for_concepts(filtered_concept_sets)

#     # Save the KG triples to a JSON file
#     with open("/shared/eng/pj20/kelpie_exp_data/kg_construct_/kg_from_pubmed.json", "w") as f:
#         json.dump(kg_triples_dict, f, indent=4)

#     print(f"Number of filtered concept sets: {len(filtered_concept_sets)}")
#     print(f"Number of concepts with triples: {len(kg_triples_dict)}")
    
    
# if __name__ == "__main__":
#     main()
    
    
import json
from typing import List, Set, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.parse import urlencode
from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response
import time
import re
import os
from tqdm import tqdm

SIMILARITY_THRESHOLD = 5

# PubMed API constants
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"
MAX_ABSTRACTS = 10
ABSTRACTS_PER_REQUEST = 5  # PubMed recommends no more than 20 IDs per request

def filter_similar_sets(concept_sets: List[Set[str]], similarity_threshold: int = 5) -> List[Set[str]]:
    """Filter out highly similar concept sets based on the concept multihot vector."""
    vectorizer = CountVectorizer(vocabulary=list(set(concept for concept_set in concept_sets for concept in concept_set)))
    concept_vectors = vectorizer.transform([' '.join(concept_set) for concept_set in concept_sets])
    filtered_sets = []
    processed_indices = set()

    for i, concept_set in enumerate(concept_sets):
        if i not in processed_indices and len(concept_set) > 10:
            similar_indices = [j for j, other_set in enumerate(concept_sets) if i != j and len(other_set) > 10 and
                               len(concept_set.symmetric_difference(other_set)) < similarity_threshold]
            filtered_sets.append(concept_set)
            processed_indices.update(similar_indices)
        elif i not in processed_indices:
            filtered_sets.append(concept_set)

    return filtered_sets

def search_pubmed(term: str, retmax: int = MAX_ABSTRACTS) -> List[str]:
    """Search PubMed for articles related to the given term."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "xml"
    }
    url = PUBMED_SEARCH_URL + "?" + urlencode(params)
    with urlopen(url) as response:
        tree = ET.parse(response)
        root = tree.getroot()
        id_list = root.find("IdList")
        return [id_elem.text for id_elem in id_list.findall("Id")]

def fetch_pubmed_abstracts(pmids: List[str]) -> List[str]:
    """Fetch abstracts for given PubMed IDs."""
    abstracts = []
    for i in range(0, len(pmids), ABSTRACTS_PER_REQUEST):
        batch = pmids[i:i+ABSTRACTS_PER_REQUEST]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract"
        }
        url = PUBMED_FETCH_URL + "?" + urlencode(params)
        with urlopen(url) as response:
            tree = ET.parse(response)
            root = tree.getroot()
            for article in root.findall(".//Article"):
                abstract = article.find(".//Abstract/AbstractText")
                if abstract is not None and abstract.text:
                    abstracts.append(abstract.text)
        time.sleep(1)  # Respect PubMed's rate limit
    return abstracts

def build_kg_for_concepts(concept_sets: List[Set[str]]) -> dict:
    """Build knowledge graph for medical concepts using PubMed abstracts and LLM."""
    kg_triples = {}

    # for concept_set in tqdm(concept_sets[:100]):
    for concept_set in tqdm(concept_sets):
        terms = list(concept_set)
        term_query = " ".join(terms)
        
        # Use PubMed API instead of dense retrieval
        pmids = search_pubmed(term_query)
        abstracts = fetch_pubmed_abstracts(pmids)

        for abstract in abstracts:
            response = extract_relationships(abstract, terms)
            triples = parse_triples(response, concept_set)

            for triple in triples:
                for concept in concept_set:
                    if concept in triple[0] or concept in triple[2]:  # Check if concept is in entity1 or entity2
                        if concept not in kg_triples:
                            kg_triples[concept] = []
                        kg_triples[concept].append(triple)
                        break

    return kg_triples


def extract_relationships(text: str, concepts: List[str]) -> str:
    """Extract relationships from text using LLM."""
    example = """Example:
Text:
Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. Common symptoms include wheezing, coughing, shortness of breath, and chest tightness. Triggers can vary but often include allergens, air pollution, exercise, and respiratory infections. Management typically involves a combination of long-term control medications, such as inhaled corticosteroids, and quick-relief medications like short-acting beta-agonists. Recent research has focused on personalized treatment approaches, including biologics for severe asthma and the role of the microbiome in asthma development and progression. Proper inhaler technique and adherence to medication regimens are crucial for effective management. Asthma action plans, developed in partnership with healthcare providers, help patients manage symptoms and exacerbations.

Concepts: [asthma, inflammation, airways, wheezing, coughing, inhaled corticosteroids, short-acting beta-agonists, allergens, respiratory infections]

Extracted triples:
[[asthma, is a, chronic respiratory condition], [asthma, characterized by, inflammation of airways], [inflammation, causes, narrowing of airways], [narrowing of airways, leads to, breathing difficulties], [wheezing, is a symptom of, asthma], [coughing, is a symptom of, asthma], [allergens, can trigger, asthma], [respiratory infections, can trigger, asthma], [inhaled corticosteroids, used for, long-term control of asthma], [short-acting beta-agonists, provide, quick relief in asthma]]
"""

    prompt = f"""Given a medical text and a list of important concepts, extract relevant relationships between the concepts from the text (if present). For each triple, if an entity matches one of the given concepts, replace the entity with the exact concept term.

Focus on generating high-quality triples closely related to the provided concepts. Aim to extract at most 10 triples for each text. Each triple should follow this format: [ENTITY1, RELATIONSHIP, ENTITY2]. Ensure the triples are informative and logically sound.

{example}

Text:
{text}

Concepts: {concepts}

Extracted triples:
"""

    response = get_claude_response(llm="sonnet", prompt=prompt)
    return response


def parse_triples(response: str, concept_set: Set[str]) -> List[Tuple[str, str, str]]:
    """Parse triples from LLM response."""
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, response)
    
    triples = []
    for match in matches:
        triple = match.split(', ')
        if len(triple) == 3:
            entity1, relation, entity2 = triple
            entity1 = entity1.replace("[", "").replace("]", "")
            entity2 = entity2.replace("[", "").replace("]", "")
            triples.append((entity1, relation, entity2))
    
    return triples

def main():

    # Load the all_visit_concepts from the JSON file
    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct/all_visit_concepts.json", "r") as f:
        all_visit_concepts = [set(visit) for visit in json.load(f)]

    if os.path.exists(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json"):
        print("Loading filtered concept sets from file...")
        with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "r") as f:
            filtered_concept_sets = json.load(f)
            # Convert lists back to sets
            filtered_concept_sets = [set(concept_set) for concept_set in filtered_concept_sets]
    else:
        print("Filtering similar concept sets...")
        filtered_concept_sets = filter_similar_sets(all_visit_concepts)

        with open(f"/shared/eng/pj20/kelpie_exp_data/kg_construct/filtered_concept_sets_{SIMILARITY_THRESHOLD}_.json", "w") as f:
            # Convert sets to lists for JSON serialization
            json.dump([list(concept_set) for concept_set in filtered_concept_sets], f)

    # Build knowledge graph for filtered concept sets
    kg_triples_dict = build_kg_for_concepts(filtered_concept_sets)

    # Save the KG triples to a JSON file
    with open("/shared/eng/pj20/kelpie_exp_data/kg_construct_/kg_from_pubmed.json", "w") as f:
        json.dump(kg_triples_dict, f, indent=4)

    print(f"Number of filtered concept sets: {len(filtered_concept_sets)}")
    print(f"Number of concepts with triples: {len(kg_triples_dict)}")
    
    
if __name__ == "__main__":
    main()