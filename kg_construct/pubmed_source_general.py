import os
import csv
from typing import Dict, List, Tuple
from tqdm import tqdm
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.parse import urlencode
import time
import re

from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response

# Constants
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"
LLM = "GPT"  # or "Claude"
MAX_ABSTRACTS = 40
ABSTRACTS_PER_REQUEST = 20  # PubMed recommends no more than 20 IDs per request
ABSTRACTS_PER_CHUNK = 5

def read_code2name(file_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Read condition, procedure, and drug mappings from CSV files."""
    condition_dict = {}
    procedure_dict = {}
    drug_dict = {}
    
    with open(f"{file_path}/CCSCM.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()
    
    with open(f"{file_path}/CCSPROC.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()
    
    with open(f"{file_path}/ATC.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['level'] == '3.0':
                drug_dict[row['code']] = row['name'].lower()
    
    return condition_dict, procedure_dict, drug_dict

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

def extract_relationships(text: str, term: str, mode: str) -> str:
    """Extract relationships from text using LLM."""
    example = get_example_for_mode(mode)
    
    prompt = f"""Given a prompt (a medical {mode}) and text summarizing multiple abstracts about it, extrapolate as many relationships as possible and provide a list of updates.
The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
Both ENTITY 1 and ENTITY 2 should be noun phrases.
Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
All relationships should be relevant to the given medical term ({term}).
Do this in both breadth and depth. Aim to extract high-quality, unique relationships from the abstracts.

{example}

Term: {term}
Text about {term}:
{text}

Updates:
"""

    if LLM == "GPT":
        response = get_gpt_response(prompt=prompt)
    elif LLM == "Claude":
        response = get_claude_response(llm="opus", prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")

    return response


def get_example_for_mode(mode: str) -> str:
    """Get an example based on the mode (condition, procedure, or drug)."""
    if mode == "condition":
        return """Example:
Term: asthma
Text about asthma:
Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to breathing difficulties. Common symptoms include wheezing, coughing, shortness of breath, and chest tightness. Triggers can vary but often include allergens, air pollution, exercise, and respiratory infections. Management typically involves a combination of long-term control medications, such as inhaled corticosteroids, and quick-relief medications like short-acting beta-agonists. Recent research has focused on personalized treatment approaches, including biologics for severe asthma and the role of the microbiome in asthma development and progression. Proper inhaler technique and adherence to medication regimens are crucial for effective management. Asthma action plans, developed in partnership with healthcare providers, help patients manage symptoms and exacerbations.

Updates: [[asthma, is a, chronic respiratory condition], [asthma, characterized by, airway inflammation], [asthma, causes, breathing difficulties], [wheezing, symptom of, asthma], [coughing, can indicate, asthma], [shortness of breath, associated with, asthma], [chest tightness, symptom of, asthma], [allergens, can trigger, asthma attacks], [air pollution, may exacerbate, asthma], [exercise, potential trigger for, asthma], [respiratory infections, can worsen, asthma], [inhaled corticosteroids, used for, long-term asthma control], [short-acting beta-agonists, provide, quick asthma relief], [biologics, treatment for, severe asthma], [microbiome, may influence, asthma development], [proper inhaler technique, important for, asthma management], [medication adherence, crucial in, asthma control], [asthma action plans, help manage, asthma exacerbations], [personalized treatment, emerging approach for, asthma], [asthma, affects, quality of life], [asthma, can be, hereditary]]
"""
    elif mode == "procedure":
        return """Example:
Term: bronchoscopy
Text about bronchoscopy:
Bronchoscopy is a medical procedure used to examine the airways of the lungs. It involves inserting a thin, flexible tube called a bronchoscope through the nose or mouth into the trachea and bronchi. The procedure allows for direct visualization of the airways, collection of samples for biopsy or culture, and therapeutic interventions. Indications for bronchoscopy include persistent cough, abnormal chest X-ray findings, suspected lung cancer, and removal of foreign objects. It can be performed under local anesthesia with sedation or general anesthesia. Recent advancements include endobronchial ultrasound (EBUS) for improved lymph node sampling and navigational bronchoscopy for accessing peripheral lung lesions. Complications, though rare, may include bleeding, pneumothorax, or respiratory distress. Bronchoscopy plays a crucial role in the diagnosis and management of various pulmonary conditions.

Updates: [[bronchoscopy, is a, medical procedure], [bronchoscopy, examines, lung airways], [bronchoscope, used in, bronchoscopy], [bronchoscopy, allows, direct airway visualization], [bronchoscopy, enables, biopsy sample collection], [bronchoscopy, can perform, therapeutic interventions], [persistent cough, indication for, bronchoscopy], [abnormal chest X-ray, may require, bronchoscopy], [bronchoscopy, used to diagnose, suspected lung cancer], [bronchoscopy, can remove, airway foreign objects], [local anesthesia, option for, bronchoscopy], [sedation, often used during, bronchoscopy], [general anesthesia, sometimes required for, bronchoscopy], [endobronchial ultrasound, advancement in, bronchoscopy], [EBUS, improves, lymph node sampling], [navigational bronchoscopy, allows access to, peripheral lung lesions], [bleeding, potential complication of, bronchoscopy], [pneumothorax, rare risk of, bronchoscopy], [respiratory distress, possible complication of, bronchoscopy], [bronchoscopy, crucial for, pulmonary condition management]]
"""
    elif mode == "drug":
        return """Example:
Term: albuterol
Text about albuterol:
Albuterol is a short-acting beta-2 adrenergic agonist commonly used in the treatment of asthma and chronic obstructive pulmonary disease (COPD). It works by relaxing smooth muscles in the airways, leading to bronchodilation and improved airflow. Albuterol is primarily administered via inhalation using a metered-dose inhaler (MDI) or nebulizer. It provides rapid relief of acute bronchospasm and is often used as a rescue medication. Common side effects include tremors, tachycardia, and nervousness. While effective for quick symptom relief, overuse of albuterol may indicate poor asthma control and the need for additional long-term control medications. Recent research has explored the use of combination inhalers containing albuterol and corticosteroids for improved asthma management. Proper inhaler technique is crucial for optimal drug delivery and efficacy.

Updates: [[albuterol, is a, short-acting beta-2 agonist], [albuterol, treats, asthma], [albuterol, used for, COPD], [albuterol, causes, bronchodilation], [albuterol, improves, airflow], [albuterol, administered via, metered-dose inhaler], [albuterol, can be given through, nebulizer], [albuterol, provides, rapid symptom relief], [albuterol, used as, rescue medication], [tremors, side effect of, albuterol], [tachycardia, potential side effect of, albuterol], [nervousness, can be caused by, albuterol], [albuterol overuse, may indicate, poor asthma control], [albuterol, relaxes, airway smooth muscles], [combination inhalers, may contain, albuterol], [albuterol, combined with, corticosteroids in some inhalers], [proper inhaler technique, important for, albuterol efficacy], [albuterol, relieves, acute bronchospasm], [albuterol, acts on, beta-2 receptors], [albuterol, improves, lung function]]
"""
    else:
        raise ValueError(f"Unknown mode: {mode}")
    

def extract_data_in_brackets(input_string: str) -> List[str]:
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches

def process_concept(term: str, mode: str, output_file: str):
    """Process a single medical concept."""
    pmids = search_pubmed(term)
    all_abstracts = fetch_pubmed_abstracts(pmids)
    
    all_relationships = []
    
    for i in range(0, min(len(all_abstracts), MAX_ABSTRACTS), ABSTRACTS_PER_CHUNK):
        chunk = all_abstracts[i:i+ABSTRACTS_PER_CHUNK]
        combined_text = "\n\n".join(chunk)
        
        response = extract_relationships(combined_text, term, mode)
        relationships = extract_data_in_brackets(response)
        all_relationships.extend(relationships)
    
    # Remove duplicates while preserving order
    unique_relationships = list(dict.fromkeys(all_relationships))
    
    outstr = ""
    for triple in unique_relationships:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(outstr)
        
    out_abstract_path = output_file.replace(".txt", "_abstracts.txt")
    with open(out_abstract_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_abstracts))

def main():
    resource_path = "./resources"
    condition_dict, procedure_dict, drug_dict = read_code2name(resource_path)
    
    graph_path = "./graphs"
    os.makedirs(graph_path, exist_ok=True)
    
    for dict_, mode, code in zip(
        [
            condition_dict, 
            procedure_dict, 
            drug_dict
            ],
        [
            "condition",
            "procedure",
            "drug"
            ],
        [
            "CCSCM", 
            "CCSPROC", 
            "ATC3"
            ]
    ):
        print(f"Generating {mode} graphs (code: {code}) from PubMed...")
        
        mode_path = os.path.join(graph_path, mode, "PubMed")
        os.makedirs(mode_path, exist_ok=True)
        
        for key, term in tqdm(dict_.items()):
            output_file = os.path.join(mode_path, f"{key}.txt")
            
            if not os.path.exists(output_file):
                process_concept(term, mode, output_file)
                time.sleep(2)  # Additional delay between processing each term
        
        print(f"Finished generating {mode} graphs (code: {code}) from PubMed")

if __name__ == "__main__":
    main()