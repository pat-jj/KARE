import h5py
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3
import heapq
from tqdm import tqdm

import torch

class AbstractRetriever:
    def __init__(self, hdf5_file, db_file, 
                 model_name="nomic-ai/nomic-embed-text-v1.5", 
                 chunk_size=10000,
                 use_cuda=False,
                 use_cosine=True):
        self.hdf5_file = hdf5_file
        self.db_file = db_file
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.use_cosine = use_cosine

        # check if cuda is available
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_cuda = True
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._load_hdf5_data()
        self.dimension = self.doc_vectors.shape[1]  # Infer dimension directly from loaded data
        self.connection = self._connect_db()
    
    def _load_hdf5_data(self):
        self.h5file = h5py.File(self.hdf5_file, 'r')
        self.doc_vectors = self.h5file['doc_vectors']
        self.doc_ids = self.h5file['doc_ids']
        self.num_rows = self.doc_vectors.shape[0]
        # self.num_rows = 1000
    
    def _connect_db(self):
        connection = sqlite3.connect(self.db_file)
        return connection
    
    def _fetch_document_info(self, pmids):
        cursor = self.connection.cursor()

        query = "SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ("
        query += ",".join([str(id) for id in pmids])
        query += ")"

        cursor.execute(query)

        rows = cursor.fetchall()
        cursor.close()
        # Convert rows to list of dictionaries
        documents = []
        for row in rows:
            documents.append({
                'pmid': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'publication_year': row[4]
            })
        return documents
    
    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return query_vector.astype('float32')
    
    def search(self, query, top_k=10):
        query_vector = self.embed_query(query)
        if self.use_cuda:
            query_vector = torch.tensor(query_vector, device=self.device)
        if self.use_cosine:
            query_norm = (query_vector ** 2).sum() ** 0.5
            query_vector = query_vector / query_norm

        similarities = -10_000. * np.ones(top_k, dtype='float32')
        indices = np.zeros(top_k, dtype='int64')

        with torch.no_grad():
            # Process in chunks to manage memory usage
            for start in tqdm(range(0, self.num_rows, self.chunk_size)):
                end = min(start + self.chunk_size, self.num_rows)
                chunk_vectors = self.doc_vectors[start:end]
                if self.use_cuda:
                    chunk_vectors = torch.tensor(chunk_vectors, device=self.device)
                chunk_indices = np.arange(start, end)

                dot_products = (chunk_vectors * query_vector[None, :]).sum(axis=1)

                if self.use_cosine:              
                    chunk_norms = (chunk_vectors ** 2).sum(axis=1) ** 0.5
                    if self.use_cuda:
                        chunk_similarities = dot_products / torch.clamp(chunk_norms, min=1e-5)
                    else:
                        chunk_similarities = dot_products / np.maximum(chunk_norms, 1e-5)
                else:
                    chunk_similarities = dot_products

                if self.use_cuda:
                    max_values, max_indices = torch.topk(chunk_similarities, top_k)
                    chunk_similarities = max_values.cpu().numpy()
                    chunk_indices = chunk_indices[max_indices.cpu()]
                else:
                    # Get indices of the largest 'top_k' elements in similarities
                    max_indices = np.argpartition(chunk_similarities, -top_k)[-top_k:]
                    chunk_similarities = chunk_similarities[max_indices]
                    chunk_indices = chunk_indices[max_indices]

                if np.sum(np.isnan(chunk_similarities)) > 0:
                    # there are nan's.
                    raise Exception("NaN found")

                # combine similarities and chunk_similarities, and indices and chunk_indices.
                # find the top-k similarities and corresponding indices.
                # set them to similarities and indices.
                all_similarities = np.concatenate([similarities, chunk_similarities])
                all_indices = np.concatenate([indices, chunk_indices])
                max_indices = np.argpartition(all_similarities, -top_k)[-top_k:]
                similarities = all_similarities[max_indices]
                indices = all_indices[max_indices]

            # sort indices according to indices
            idx = np.argsort(indices)
            indices = indices[idx]
            similarities = similarities[idx]

            pmids = self.doc_ids[indices]

            documents = self._fetch_document_info(pmids)

        return pmids, similarities, documents

# Example usage
if __name__ == "__main__":
    hdf5_file = 'data.h5'
    db_file = 'articles.db'

    retriever = AbstractRetriever(hdf5_file, db_file)
    query = "example search query"
    similarities, results = retriever.search(query, top_k=10)

    print("Top K results:", results)

