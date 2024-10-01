import numpy as np
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
from sklearn.metrics import silhouette_score
import json
import pickle
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


EMB_DIMENSIONS = 1024
ENT_THRESHOLD = 0.14
REL_THRESHOLD = 0.14

# Load the API key
with open('./apis/openai.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

def read_knowledge_graph(path):
    entities = set()
    relations = set()
    with open(path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
    return list(entities), list(relations)

def generate_embeddings(texts, model="text-embedding-3-large", num_threads=30):
    embeddings = {}

    def process_text(text):
        embedding = client.embeddings.create(
            input=text,
            model=model,
            dimensions=EMB_DIMENSIONS,
        ).data[0].embedding
        return text, embedding

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_text, text) for text in texts]
        for future in tqdm(as_completed(futures), total=len(texts)):
            text, embedding = future.result()
            embeddings[text] = embedding

    return embeddings

def find_optimal_threshold(embeddings, min_threshold=0.05, max_threshold=0.37, num_thresholds=32, sample_size=40000, specified_threshold=None):
    if specified_threshold is None:
        thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
        best_score = -1.0
        best_threshold = None
        best_labels = None

        # Randomly select a sample of entity names
        sample_entities = np.random.choice(list(embeddings.keys()), size=min(sample_size, len(embeddings)), replace=False)
        sample_embeddings = [embeddings[entity] for entity in sample_entities]

        def evaluate_threshold(threshold):
            print(f"Evaluating threshold: {threshold:.2f}")
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average', metric='cosine')
            labels = clustering.fit_predict(sample_embeddings)
            score = silhouette_score(sample_embeddings, labels)

            # Create mappings for the current threshold
            new_to_original_mapping, _ = create_mappings(labels, sample_entities, embeddings)

            # Save the new_to_original mapping for the current threshold
            output_dir = "/shared/eng/pj20/kelpie_exp_data/cluster_test"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/new_to_original_mapping_threshold_{threshold:.2f}.json", 'w') as f:
                json.dump(new_to_original_mapping, f, indent=4)

            return threshold, score, labels

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_threshold, threshold) for threshold in thresholds]
            for future in as_completed(futures):
                threshold, score, labels = future.result()
                logging.info(f"Threshold: {threshold:.2f}, Silhouette Score: {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_labels = labels
        print(f"Best threshold: {best_threshold:.2f}, Best Silhouette Score: {best_score:.3f}.")
    else:
        best_threshold = specified_threshold

    print(f"Now evaluating the best threshold on the full dataset with threshold: {best_threshold:.2f}")
    # Assign labels to the full dataset using the best threshold
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, linkage='average', metric='cosine')
    best_labels = clustering.fit_predict(list(embeddings.values()))

    return best_threshold, best_labels, clustering

def find_cluster_representative(cluster_items, embeddings_dict):
    cluster_embeddings = [embeddings_dict[item] for item in cluster_items]
    cluster_center = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
    representative_index = np.argmin(distances)
    return cluster_items[representative_index]

def create_mappings(labels, items, embeddings_dict):
    new_to_original_mapping = {}
    original_to_new_mapping = {}

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_items = [items[idx] for idx in cluster_indices]
        representative_item = find_cluster_representative(cluster_items, embeddings_dict)
        new_to_original_mapping[representative_item] = cluster_items
        for item in cluster_items:
            original_to_new_mapping[item] = representative_item

    return new_to_original_mapping, original_to_new_mapping

def compute_cluster_embeddings(mapping, embeddings):
    cluster_embeddings = {}
    for new_item, original_items in mapping.items():
        cluster_embeddings[new_item] = np.mean([embeddings[item] for item in original_items], axis=0).tolist()
    return cluster_embeddings

def save_cluster_embeddings(cluster_embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(cluster_embeddings, f)

def main():
    kg_path = "../graph/kg_theme_raw.txt"
    output_dir = "/shared/eng/pj20/kelpie_exp_data/clustering_theme"
    num_threads = 15

    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'clustering.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Suppress logging for the openai module
    logging.getLogger('openai').setLevel(logging.WARNING)

    logging.info("Reading knowledge graph data...")
    entities, relations = read_knowledge_graph(kg_path)

    # Check if embeddings exist, if not generate them
    entity_embeddings_file = f"{output_dir}/entity_embeddings_{EMB_DIMENSIONS}.pkl"
    if os.path.exists(entity_embeddings_file):
        logging.info(f"Loading entity embeddings from {entity_embeddings_file}...")
        with open(entity_embeddings_file, 'rb') as f:
            entity_embeddings = pickle.load(f)
    else:
        logging.info(f"Generating embeddings for {len(entities)} entities using {num_threads} threads...")
        # Temporarily change logging level to WARNING
        logging.getLogger().setLevel(logging.WARNING)
        entity_embeddings = generate_embeddings(entities, num_threads=num_threads)
        # Restore logging level to INFO
        logging.getLogger().setLevel(logging.INFO)
        with open(entity_embeddings_file, 'wb') as f:
            pickle.dump(entity_embeddings, f)

    relation_embeddings_file = f"{output_dir}/relation_embeddings_{EMB_DIMENSIONS}.pkl"
    if os.path.exists(relation_embeddings_file):
        logging.info(f"Loading relation embeddings from {relation_embeddings_file}...")
        with open(relation_embeddings_file, 'rb') as f:
            relation_embeddings = pickle.load(f)
    else:
        logging.info(f"Generating embeddings for {len(relations)} relations using {num_threads} threads...")
        # Temporarily change logging level to WARNING
        logging.getLogger().setLevel(logging.WARNING)
        relation_embeddings = generate_embeddings(relations, num_threads=num_threads)
        # Restore logging level to INFO
        logging.getLogger().setLevel(logging.INFO)
        with open(relation_embeddings_file, 'wb') as f:
            pickle.dump(relation_embeddings, f)

    if os.path.exists(f"{output_dir}/entity_labels_{ENT_THRESHOLD}.pkl"):
        logging.info("Loading existing entity labels...")
        with open(f"{output_dir}/entity_labels_{ENT_THRESHOLD}.pkl", 'rb') as f:
            entity_labels = pickle.load(f)
        entity_threshold = ENT_THRESHOLD
    else:
        logging.info("Finding optimal threshold for entity clustering...")
        entity_threshold, entity_labels, entity_clustering_res = find_optimal_threshold(entity_embeddings, specified_threshold=0.14)
        logging.info(f"Optimal threshold for entity clustering: {entity_threshold:.2f}")
        with open(f"{output_dir}/entity_labels_{entity_threshold}.pkl", 'wb') as f:
            pickle.dump(entity_labels, f)
        with open(f"{output_dir}/entity_clustering_res_{entity_threshold}.pkl", 'wb') as f:
            pickle.dump(entity_clustering_res, f)

    logging.info(f"Creating mappings between new and original entities...")
    entities_ = list(entity_embeddings.keys())
    new_to_original_entity_mapping, original_to_new_entity_mapping = create_mappings(entity_labels, entities_, entity_embeddings)

    logging.info("Saving entity mappings...")
    with open(f"{output_dir}/new_to_original_entity_mapping_{entity_threshold}.json", 'w') as f:
        json.dump(new_to_original_entity_mapping, f, indent=4)

    with open(f"{output_dir}/original_to_new_entity_mapping_{entity_threshold}.json", 'w') as f:
        json.dump(original_to_new_entity_mapping, f, indent=4)

    if os.path.exists(f"{output_dir}/relation_labels_{REL_THRESHOLD}.pkl"):
        logging.info("Loading existing relation labels...")
        with open(f"{output_dir}/relation_labels_{REL_THRESHOLD}.pkl", 'rb') as f:
            relation_labels = pickle.load(f)
        relation_threshold = REL_THRESHOLD
    else:
        logging.info("Finding optimal threshold for relation clustering...")
        relation_threshold, relation_labels, relation_clustering_res = find_optimal_threshold(relation_embeddings, specified_threshold=0.14)
        logging.info(f"Optimal threshold for relation clustering: {relation_threshold:.2f}")
        with open(f"{output_dir}/relation_labels_{relation_threshold}.pkl", 'wb') as f:
            pickle.dump(relation_labels, f)
        with open(f"{output_dir}/relation_clustering_res_{relation_threshold}.pkl", 'wb') as f:
            pickle.dump(relation_clustering_res, f)

    logging.info(f"Creating mappings between new and original relations...")
    relations_ = list(relation_embeddings.keys())
    new_to_original_relation_mapping, original_to_new_relation_mapping = create_mappings(relation_labels, relations_, relation_embeddings)

    logging.info("Saving relation mappings...")
    with open(f"{output_dir}/new_to_original_relation_mapping_{relation_threshold}.json", 'w') as f:
        json.dump(new_to_original_relation_mapping, f, indent=4)

    with open(f"{output_dir}/original_to_new_relation_mapping_{relation_threshold}.json", 'w') as f:
        json.dump(original_to_new_relation_mapping, f, indent=4)

    logging.info("Computing and saving entity cluster embeddings...")
    entity_cluster_embeddings = compute_cluster_embeddings(new_to_original_entity_mapping, entity_embeddings)
    save_cluster_embeddings(entity_cluster_embeddings, f"{output_dir}/refined_entity_embeddings.pkl")

    logging.info("Computing and saving relation cluster embeddings...")
    relation_cluster_embeddings = compute_cluster_embeddings(new_to_original_relation_mapping, relation_embeddings)
    save_cluster_embeddings(relation_cluster_embeddings, f"{output_dir}/refined_relation_embeddings.pkl")

    # Check if mappings exist, if not update the knowledge graph
    new_kg_path = kg_path.replace('kg_theme_raw', 'kg_theme_refined')
    new_kg_triples = set()
    if not os.path.exists(new_kg_path):
        logging.info("Updating knowledge graph with new entity and relation names...")
        with open(new_kg_path, 'w') as f:
            with open(kg_path, 'r') as kg_file:
                for line in kg_file:
                    head, relation, tail = line.strip().split('\t')
                    new_head = original_to_new_entity_mapping[head]
                    new_tail = original_to_new_entity_mapping[tail]
                    new_relation = original_to_new_relation_mapping[relation]
                    triple = f"{new_head}\t{new_relation}\t{new_tail}\n"
                    if triple not in new_kg_triples:
                        new_kg_triples.add(triple)
                        f.write(triple)

    logging.info("Clustering completed successfully.")

if __name__ == "__main__":
    main()