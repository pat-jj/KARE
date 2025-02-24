# Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval


### 1. Prepare EHR data
```bash
cd ehr_prepare
python ehr_data_prepare.py
python sample_prepare.py
```
#### (Baseline Models Evaluation)
```bash
baselines/baseline_play.ipynb
```

### 2. Knowledge Graph (KG) Construction

**General Preparation:**
```bash
cd kg_construct
python query_data_prepare.py
```

**Extract KG from PubMed:**

Preparation:
```bash
cd kg_construct/pubmed_index
python download_pubmed.py
python embed_pubmed.py
python convert_dat.py
```

Construct KG:
```bash
cd kg_construct
python pubmed_source.py
```


**Extract KG from UMLS:**

Source data files: [graph.txt](https://storage.googleapis.com/pyhealth/umls/graph.txt) (UMLS KG) and [UMLS.csv](https://storage.googleapis.com/pyhealth/umls/UMLS.csv) (mapping file)

```bash
cd kg_construct
python umls_source.py
```

Alternatively, you can use our processed UMLS KG: [Google Drive](https://drive.google.com/file/d/1Zs4hXUiXs_ikkHjHbqp9ZEoH4l6WEP5H/view?usp=sharing)

**Extract KG from LLM:**
```bash
cd kg_construct
python llm_source.py
```

**Semantic Clustering:**

After combining all the KGs into kg_raw.txt under "graph" folder (in project root), run:
```bash
cd kg_construct
python refine_kg.py
```

### 3. KG Community Detection and Indexing (Summarization)
```bash
cd kg_index
python structure_partition_leiden.py
```


### 4. Patient Context Construction and Augmentation
```bash
cd patient_context
python base_context.py
python get_emb.py
python sim_patient_ret_faiss.py
python augment_context.py
```

### 5. Reasoning Chain Generation 
```bash
cd prediction
python data_prepare.py
python split_task.py

```


### 6. LLM Fine-tuning
Please follow https://github.com/huggingface/alignment-handbook to build the environment for fine-tuning.
Start the fine-tuning for the specific task (mortality/readmission):
```bash
sh finetune/sft_{task}.sh
```

### 7. Prediction & Evaluation
```bash
# For the prediction
cd prediction
cd llm_inference
python generate.py

# For the evaluation
cd prediction
python eval.py
```

### * A Cost-Effective/Naive Approach (Skipping Step 1-3) to Validate Our Results
---
This approach will directly retrieve the knowledge summaries from an LLM, and use them to construct the input and output for LLM fine-tuning. However, the result would not be as good as the original method, but can still be used to validate the philosophy underlying our method.

    (This approach is suitable for those who do not want to spend money on building their own context-aware and concept-specific KG.)

    **Major Advantange** over our method: 
    (1) Much lower cost than our original implementation.
    (2) No need to tune the hyperparameters for the context augmentation.

    **Major Disadvantage**: 
    (1) Relatively lower performance as it only uses the knowledge from LLM. 
    (2) For real-world application, you will need to prepare the knowledge from the **same LLM** for the every new sample during the inference -> higher cost if the application is long-term.
    
```bash
cd prediction
python dp_new.py
```

Others
---
To call LLM APIs in this work, you need to 

Enter AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION in ``apis_example/claude_api.py`` to call Claude APIs.

Enter your OpenAI API key in ``apis_example/openai.key`` to call OpenAI APIs.

Then, you need to rename ``apis_example`` to ``apis``, and put it under each folder where you need to call APIs.

---

## Cite KARE
```bibtex
@misc{jiang2024reasoningenhancedhealthcarepredictionsknowledge,
      title={Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval}, 
      author={Pengcheng Jiang and Cao Xiao and Minhao Jiang and Parminder Bhatia and Taha Kass-Hout and Jimeng Sun and Jiawei Han},
      year={2024},
      eprint={2410.04585},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.04585}, 
}
```

Thank you for your interest in our work!