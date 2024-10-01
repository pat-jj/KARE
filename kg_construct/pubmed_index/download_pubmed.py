import ftplib
import os
import gzip
import xml.etree.ElementTree as ET
import sqlite3
from tqdm import tqdm

# Define constants
FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_DIR = "/pubmed/baseline/"
LOCAL_DIR = "/shared/eng/pj20/kelpie_exp_data"
PLACEHOLDER_DIR = "pubmed"
DB_FILE = "pubmed_data.db"

# Ensure local directories exist
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(PLACEHOLDER_DIR, exist_ok=True)

# Connect to the FTP server and list files
ftp = ftplib.FTP(FTP_HOST)
ftp.login()
ftp.cwd(FTP_DIR)
files = ftp.nlst()

# Filter files to only include .xml or .xml.gz
files = [file for file in files if file.endswith('.xml') or file.endswith('.xml.gz')]

# Connect to SQLite database
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        pmid TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        abstract TEXT,
        publication_year INTEGER
    )
''')

# Function to extract article data from XML files
def extract_article_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            abstract = article.findtext(".//AbstractText")
            pub_year = article.findtext(".//PubDate/Year")
            if not pub_year:
                pub_year = article.findtext(".//PubDate/MedlineDate")
                if pub_year and len(pub_year) >= 4:
                    pub_year = pub_year[:4]
            
            authors_list = []
            for author in article.findall(".//Author"):
                last_name = author.findtext("LastName")
                fore_name = author.findtext("ForeName")
                if last_name and fore_name:
                    authors_list.append(f"{fore_name} {last_name}")
            authors = ", ".join(authors_list)

            if pmid and (title or abstract):
                yield pmid, title, authors, abstract, pub_year

# Download, extract, and save article data
for file in tqdm(files, desc="Downloading and processing files", unit="file"):
    placeholder_file = os.path.join(PLACEHOLDER_DIR, f"{file}.processed")
    
    if os.path.exists(placeholder_file):
        print(f"Skipping {file}, already processed.")
        continue

    local_file = os.path.join(LOCAL_DIR, file)
    if not os.path.exists(local_file):
        with open(local_file, 'wb') as f:
            ftp.retrbinary(f"RETR {file}", f.write)
    
    for pmid, title, authors, abstract, pub_year in extract_article_data(local_file):
        cur.execute('INSERT OR IGNORE INTO articles (pmid, title, authors, abstract, publication_year) VALUES (?, ?, ?, ?, ?)', 
                    (pmid, title, authors, abstract, pub_year))
    conn.commit()

    # Remove the processed file to free up space
    os.remove(local_file)
    
    # Create a placeholder file to indicate this file has been processed
    open(placeholder_file, 'a').close()

# Close connections
ftp.quit()
conn.close()

print("All files processed and article data saved to the database.")