import os
import csv
from typing import Dict, List, Tuple
from tqdm import tqdm
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.parse import urlencode
import time
import re
import pickle

from apis.gpt_api import get_gpt_response
from apis.claude_api import get_claude_response

# Constants
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"
LLM = "Claude"  # or "Claude"
MAX_ABSTRACTS = 30
ABSTRACTS_PER_REQUEST = 10  # PubMed recommends no more than 20 IDs per request
ABSTRACTS_PER_CHUNK = 5
LUNG_CANCER_TERM = "lung cancer"

def read_code2name(file_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Read condition, procedure, and drug mappings from CSV files."""
    condition_dict = {}
    procedure_dict = {}
    drug_dict = {}
    
    with open(f"{file_path}/CCSCM.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()
            
    condition_dict['19'] = "Lung Cancer"
    
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
    """Search PubMed for articles related to the given term and lung cancer."""
    params = {
        "db": "pubmed",
        "term": f"({term}) AND ({LUNG_CANCER_TERM})",  # Search for both the term and lung cancer
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
    
    prompt = f"""Given a term (a medical {mode}) and text summarizing multiple abstracts about its relationship with lung cancer, extrapolate as many relationships as possible and provide a list of updates.
The relationships should highlight direct or indirect connections between the {mode} ({term}) and lung cancer.
Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
The relationships should be relevant and potentially helpful to healthcare prediction, focusing on lung cancer mortality and risk factors.
Both ENTITY 1 and ENTITY 2 should be noun phrases.
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
        response = get_claude_response(llm="sonnet", prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM: {LLM}")

    return response


def get_example_for_mode(mode: str) -> str:
    """Get an example based on the mode (condition, procedure, or drug)."""
    if mode == "condition":
        return """Example:
Term: COPD
Text about COPD:
Chronic obstructive pulmonary disease (COPD) and lung cancer are closely related respiratory conditions. COPD, characterized by airflow limitation and chronic inflammation, is a significant risk factor for the development of lung cancer. Shared risk factors, such as smoking and air pollution, contribute to the pathogenesis of both diseases. The chronic inflammation and oxidative stress present in COPD create a favorable environment for carcinogenesis. Patients with COPD have a higher incidence of lung cancer compared to the general population, with the risk increasing with the severity of COPD. The presence of emphysema, a subtype of COPD, is particularly associated with an increased risk of lung cancer. Early detection and management of COPD may have implications for lung cancer screening and prevention strategies. Studies have shown that COPD is associated with poorer prognosis and increased mortality in lung cancer patients. The coexistence of COPD and lung cancer poses challenges in terms of diagnosis, treatment, and overall patient outcomes.

Updates: [[COPD, is a significant risk factor for, lung cancer development], [smoking, shared risk factor for, COPD and lung cancer], [air pollution, contributes to, COPD and lung cancer pathogenesis], [chronic inflammation in COPD, creates favorable environment for, lung cancer carcinogenesis], [oxidative stress in COPD, promotes, lung cancer development], [COPD severity, correlates with, increased lung cancer risk], [emphysema, particularly associated with, increased lung cancer risk], [early COPD detection, may inform, lung cancer screening strategies], [COPD management, potential implications for, lung cancer prevention], [COPD, associated with, poorer lung cancer prognosis], [COPD, linked to, increased lung cancer mortality], [coexistence of COPD and lung cancer, poses challenges in, diagnosis and treatment], [COPD presence, impacts, overall lung cancer patient outcomes]]
"""
    elif mode == "procedure":
        return """Example:
Term: lung biopsy
Text about lung biopsy:
Lung biopsy is a diagnostic procedure used to obtain tissue samples from the lungs for pathological examination. It plays a crucial role in the diagnosis and staging of lung cancer. When imaging studies, such as chest X-rays or CT scans, reveal suspicious lung nodules or masses, a lung biopsy is often performed to confirm the presence of malignancy. Different types of lung biopsies include transthoracic needle biopsy, bronchoscopic biopsy, and surgical biopsy. The choice of biopsy technique depends on the location and size of the lesion, as well as patient factors. Lung biopsy provides tissue samples for histological and molecular analysis, enabling the determination of lung cancer type and guiding personalized treatment decisions. Complications of lung biopsy, although rare, may include pneumothorax, bleeding, or infection. Advances in image-guided biopsy techniques and molecular testing have improved the diagnostic accuracy and therapeutic relevance of lung biopsies in the management of lung cancer. Timely and accurate lung biopsy results are essential for initiating appropriate treatment and assessing prognosis in lung cancer patients. Delayed or inconclusive biopsy results may negatively impact survival outcomes.

Updates: [[lung biopsy, used for, lung cancer diagnosis and staging], [suspicious lung nodules, may require, lung biopsy for confirmation], [lung masses, often investigated with, lung biopsy], [transthoracic needle biopsy, type of, lung biopsy], [bronchoscopic biopsy, performed for, lung cancer diagnosis], [surgical biopsy, option for, lung cancer diagnosis], [lung biopsy, provides tissue for, histological and molecular analysis], [lung biopsy results, guide, personalized lung cancer treatment decisions], [pneumothorax, potential complication of, lung biopsy], [bleeding, rare complication of, lung biopsy], [image-guided biopsy, improves, lung cancer diagnostic accuracy], [molecular testing of lung biopsies, enhances, therapeutic relevance in lung cancer], [timely lung biopsy results, essential for, initiating appropriate lung cancer treatment], [accurate lung biopsy results, crucial for, assessing lung cancer prognosis], [delayed lung biopsy results, may negatively impact, lung cancer survival outcomes], [inconclusive lung biopsy results, can hinder, lung cancer management]]
"""
    elif mode == "drug":
        return """Example:
Term: cisplatin
Text about cisplatin:
Cisplatin is a chemotherapeutic agent widely used in the treatment of lung cancer. It is a platinum-based drug that works by inducing DNA damage and apoptosis in cancer cells. Cisplatin is often used in combination with other chemotherapy drugs as a first-line treatment for advanced non-small cell lung cancer (NSCLC) and extensive-stage small cell lung cancer (SCLC). It has demonstrated efficacy in improving survival outcomes and reducing tumor burden. However, cisplatin is associated with significant side effects, including nausea, vomiting, nephrotoxicity, and peripheral neuropathy. Strategies such as hydration and antiemetic prophylaxis are employed to manage these adverse effects. Resistance to cisplatin can develop over time, leading to treatment failure. Research efforts are focused on identifying biomarkers to predict cisplatin response and developing novel drug combinations to overcome resistance. Cisplatin-based chemotherapy remains a cornerstone in the management of lung cancer, often used in conjunction with targeted therapies and immunotherapies to improve patient outcomes. The optimal use of cisplatin in lung cancer treatment involves careful patient selection, dosing, and monitoring to maximize efficacy while minimizing toxicity. Cisplatin-related toxicities may impact patient quality of life and adherence to treatment, ultimately affecting survival outcomes.

Updates: [[cisplatin, is a chemotherapeutic agent for, lung cancer treatment], [cisplatin, induces DNA damage in, lung cancer cells], [cisplatin, promotes apoptosis in, lung cancer cells], [cisplatin, used as first-line treatment for, advanced NSCLC], [cisplatin, combined with other drugs for, lung cancer treatment], [cisplatin, improves survival outcomes in, lung cancer patients], [cisplatin, reduces tumor burden in, lung cancer], [nausea and vomiting, common side effects of, cisplatin in lung cancer treatment], [nephrotoxicity, potential complication of, cisplatin therapy], [peripheral neuropathy, can occur with, cisplatin treatment], [hydration and antiemetic prophylaxis, strategies to manage, cisplatin side effects], [cisplatin resistance, can lead to, lung cancer treatment failure], [biomarkers, being studied to predict, cisplatin response in lung cancer], [novel drug combinations, explored to overcome, cisplatin resistance], [cisplatin, often combined with targeted therapies for, improved lung cancer outcomes], [cisplatin, used with immunotherapies for, enhanced lung cancer treatment], [optimal cisplatin use, involves careful patient selection and monitoring], [cisplatin toxicities, may impact, lung cancer patient quality of life], [cisplatin-related side effects, can affect, treatment adherence and survival outcomes]]
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
    lung_mort_dataset_path = "/shared/eng/pj20/kelpie_exp_data/ehr_data/mimic3_lung_mortality.pkl"
    with open(lung_mort_dataset_path, "rb") as f:
        lung_mort_dataset = pickle.load(f)
        
    samples = [sample for sample in lung_mort_dataset.samples if sample['label'] != 1000]
    lung_mort_dataset.samples = samples
    
    code_to_handle = {
        "condition": lung_mort_dataset.get_all_tokens('conditions'),
        "procedure": lung_mort_dataset.get_all_tokens('procedures'),
        "drug": lung_mort_dataset.get_all_tokens('drugs')
    }
    
    graph_path = "./theme_specific_graphs"
    # os.makedirs(graph_path, exist_ok=True)
    
    for dict_, mode, code in zip(
        [
            condition_dict, 
            # procedure_dict, 
            # drug_dict
            ],
        [
            "condition",
            # "procedure",
            # "drug"
            ],
        [
            "CCSCM", 
            # "CCSPROC", 
            # "ATC3"
            ]
    ):
        print(f"Generating {mode} graphs (code: {code}) from PubMed...")
        
        mode_path = os.path.join(graph_path, mode, "PubMed")
        os.makedirs(mode_path, exist_ok=True)
        
        for key, term in tqdm(dict_.items()):
            output_file = os.path.join(mode_path, f"{key}.txt")
            
            if not os.path.exists(output_file) and key in code_to_handle[mode]:
                process_concept(term, mode, output_file)
                time.sleep(2)
        
        print(f"Finished generating {mode} graphs (code: {code}) from PubMed")

if __name__ == "__main__":
    main()