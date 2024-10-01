from abstract_retriever import AbstractRetriever
import os

db_file = "pubmed_data.db"
h5_file = "pubmed_embeddings.h5"

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
retriever = AbstractRetriever(h5_file, db_file, chunk_size=250000, use_cuda=True)

while True:

    # Example query embedding (replace with an actual embedding)
    query = input("Enter your query: ").strip()

    if query == "":
        query = """Patient ID: 29488_1

Visit 0:
Conditions:
- Deficiency and other anemia
- Essential hypertension
- Complication of device; implant or graft
- Congestive heart failure; nonhypertensive
- Cancer of prostate
- Anxiety disorders
- Thyroid disorders
- Disorders of lipid metabolism
- Conduction disorders
- Mycoses
- Other diseases of kidney and ureters
- Cancer of esophagus

Procedures:
- Diagnostic cardiac catheterization; coronary arteriography
- Other or procedures on vessels other than head and neck
- Colorectal resection

Drugs:
- Other drugs for obstructive airway diseases, inhalants in ATC
- Lipid modifying agents, plain
- Antithrombotic agents
- Angiotensin II receptor blockers (ARBs), plain

Visit 1:
Conditions:
- Congestive heart failure; nonhypertensive
- Cardiac dysrhythmias
- Shock
- Alcohol-related disorders
- Diabetes mellitus with complications
- E codes: adverse effects of medical care
- Disorders of teeth and jaw

Procedures:
- Other vascular catheterization; not heart
- Blood transfusion

Drugs:
- Other drugs for obstructive airway diseases, inhalants in ATC
- Other analgesics and antipyretics in ATC
- Drugs for peptic ulcer and gastro-oesophageal reflux disease (GORD)
- Beta blocking agents
- Potassium supplements
"""

    if query.lower() == "exit":
        break

    print(f"Looking up the PubMed abstracts using \"{query}\"...")

    print("Embedding the query...")
    query_embedding = retriever.embed_query(query)
    print("Query embedded.")

    print("Searching for similar abstracts...")
    top_k = 10
    pmids, distances, documents = retriever.search(query, top_k)
    print("Search completed.")

    # sort documents according to distances (descending)
    distances, documents = zip(*sorted(zip(distances, documents), key=lambda x: x[0], reverse=True))

    for i, (abstract, similarity) in enumerate(zip(documents, distances)):
        print(f"Rank {i + 1}, Similarity: {similarity}")
        print(f"PMID: {abstract['pmid']}")
        print(f"Title: {abstract['title']}")
        print(f"Authors: {abstract['authors']}")
        print(f"Abstract: {abstract['abstract']}")
        print(f"Publication Year: {abstract['publication_year']}")
        print("-----")






