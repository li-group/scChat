# scChat: A Large Language Model-Powered Co-Pilot for Contextualized Single-Cell RNA Sequencing Analysis
Welcome to the [scChat](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) page. scChat is a pioneering AI assistant designed to enhance single-cell RNA sequencing (scRNA-seq) analysis by incorporating research context into the workflow. Powered by a large language model (LLM), scChat goes beyond standard tasks like cell annotation by offering advanced capabilities such as research context-based experimental analysis, hypothesis validation, and suggestions for future experiments.

### Video Demo
Watch the demo of scChat in action below:

[![scChat Video Demo](https://img.youtube.com/vi/4LDdncq-sp8/0.jpg)](https://youtu.be/4LDdncq-sp8)

If you found this work useful, please cite this [preprint](https://arxiv.org/abs/2308.12923) as:
```bibtex
@misc{lu2024scchat,
    title={scChat: A Large Language Model-Powered Co-Pilot for Contextualized Single-Cell RNA Sequencing Analysis},
    author={Yen-Chun Lu and Ashley Varghese and Rahul Nahar and Hao Chen and Kunming Shao and Xiaoping Bao and Can Li},
    year={2024},
    eprint={2024.10.01.616063},
    archivePrefix={bioRxiv},
    doi={10.1101/2024.10.01.616063}
}
```

## Table of Contents
- [Overview](#overview)
- [Tutorial](#tutorial)
- [Chat Example](#chat-example)
- [Datasets](#datasets)
- [Citation](#citation)

# Overview
<a name="overview"></a>
## Motivation
Data-driven methods such as unsupervised and supervised learning are essential tools in single-cell RNA sequencing (scRNA-seq) analysis. However, these methods often lack the ability to incorporate research context, which can lead to overlooked insights. scChat addresses this by integrating contextualized conversation with data analysis to provide a deeper understanding of experimental results. It supports the exploration of research hypotheses and generates actionable insights for future experiments.

Please read our [scChat paper](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) for more motivation and details about how the scChat works.

## Scope
Model: scChat currently supports analysis using AnnData-formatted single-cell RNA sequencing datasets.

Capabilities: scChat integrates an LLM mutli-agent system with specialized tools to enable tasks, such as cell type annotation, enrichment analysis, and result visualization, all through conversational interactions.


# Tutorial 

To set up the project environment and run the server, follow these steps:

1. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt

Follow these steps to utilize the application effectively:
### Step 1: Set the OPENAI Key Environment Variable 
- Type and enter export OPENAI_API_KEY='your_openai_api_key' in your terminal

  
### Step 2: Download Neo4j Desktop 2
- Download Neo4j Desktop 2 (https://neo4j.com/download/)
- Download required dump files (https://drive.google.com/drive/folders/17UCKv95G3tFqeyce1oQAo3ss2vS7uZQE)
- Create a new instance on Neo4j (this step asks you set the password)
- Import the dump files as new databases in the created instance.
- Start the database

### Step 3: Upload and update files
- Upload scRNA-seq adata file (.h5ad)
- Upload the pathway vector-based model (.pkl and .faiss), which can be found in this link: https://drive.google.com/drive/u/4/folders/1OklM2u5T5FsjiUvvYRYyWxrssQIb84Ky
- Update specification_graph.json with your Neo4j username, password, system and organ relevant to the database you are using with specific format
- Update sample_mapping.json with adata file corresponding "Sample name", which can be found in adata.obs, and write descriptions for each condition.

  
### Step 4: Initialize the Application
- Run python3 manage.py migrate (For the first time as you install scChat)
- Run python3 manage.py runserver

### Step 5: Access the Application
- Open your web browser and navigate to:
  `http://127.0.0.1:8000/schatbot`
  
- **Recommended:** Use Google Chrome for the best experience.

### Additional Tip: Clear Cache to Avoid Previous Chat Data
- Periodically clearing the cache is recommended to ensure a smooth experience:
  1. Right-click on the page and select **Inspect**.
  2. Go to the **Application** tab.
  3. Under **Cookies**, remove `sessionid`.
  
  This will prevent previous chat sessions from being reprocessed.

## Chat Example
<a name="chat-example"></a>
<p align="center">
<!-- <img src="images/Chatbot_eg_highPPI.png" alt="drawing" width="700"/> -->
</p>

# Datasets
The datasets used for testing and examples for sample_mapping.json and specification_graph.json can be found at [https://docs.google.com/spreadsheets/d/1NwN5GydHn0B3-W0DLcAfvnNtZVJEMUgBW9YyzXnS83A/edit?usp=sharing
](https://drive.google.com/drive/u/4/folders/1RJRETtwI3zxsOJK0Lop197JGm3Isl4iB)

## scChat Retrieval-Augmented Generation (RAG) Configuration

The `cell type RAG` allows users to specify the system and organ for their single-cell RNA sequencing (scRNA-seq) analysis. Based on the source of the adata the RAG database can be set as human or mouse.

### Usage
Users need to configure the `.json` file by specifying the organism (`human` or `mouse`) in the database part.

For example, to analyze peripheral blodd from lymphatic system, the configuration would look like this:

```json
{
    "url": "put your url here", 
    "username": "put your username here",
    "password": "put your password here",
    "database": "make sure the database name is correct",
    "pathway_rag": "make sure the pathway rag name is correct",
    "sources": [
        {
            "system": "Lymphatic System",
            "organ": "Peripheral blood"
        }
    ]
}
```
It's also allowed to pass multiple system and organ to the RAG. For example:
```json
{
    "url": "put your url here", 
    "username": "put your username here",
    "password": "put your password here",
    "database": "make sure the database name is correct",
    "pathway_rag": "make sure the pathway rag name is correct",
    "sources": [
        {
            "system": "Lymphatic System",
            "organ": "Peripheral blood"
        },
        {
            "system": "Nervous System",
            "organ": "Brain"
        }
    ]
}
```

This configuration will retrieve and process the relevant dataset based on the selected system and organ.

### Available systems and organs

#### Human
- Nervous system
  - Brain
  - Spinal cord
  - Ganglion
  - Eye
  - Ear     
- Musculoskeletal system
  - Synovial tissue
  - Tendo
  - Muscle
  - Bone
- Female Reproductive System
  - Vagina
  - Cervix
  - Egg
  - Breast
  - Uterus
  - Ovary
  - Oviduct (Fallopian tube)
- Cardiovascular System
  - Capillary
  - Heart
  - Artery
  - Vein
- Digestive System
  - Stomach
  - Esophagus
  - Liver
  - Mouth
  - Intestine
  - Abdomen
- Endocrine System
  - Pancreas
  - Parathyroid gland
  - Adrenal gland
  - Thyroid gland
- Urinary System
  - Kidney
  - Ureters
  - Bladder
  - Urethra
- Male Reproductive System
  - Penis
  - Testis
  - Prostate gland
- Respiratory System
  - Nose
  - Lung
  - Trachea
  - Larynx
  - Pharynx
- Lymphatic System
  - Spleen
  - Peripheral blood
  - Umbilical cord blood
  - Bone marrow
  - Lymph node
  - Thymus
- Integumentary System
  - Adipose tissue
  - Skin
- Embryonic Structure
  - Embryo
  - Fetus
  - Placenta

#### Mouse
- Endocrine System
  - Thyroid gland
  - Parathyroid gland
  - Adrenal gland
  - Pancreas
- Cardiovascular System
  - Capillary
  - Artery
  - Vein
  - Heart
- Nervous System
  - Spinal cord
  - Eye
  - Ear
  - Ganglion
  - Brain
- Respiratory System
  - Nose
  - Larynx
  - Trachea
  - Lung
  - Pharynx
- Integumentary System
  - Adipose tissue
  - Skin
- Lymphatic System
  - Thymus
  - Peripheral blood
  - Bone marrow
  - Spleen
  - Lymph node
- Embryonic Structure
  - Embryo
  - Placenta
  - Fetus
- Female Reproductive System
  - Oviduct (Fallopian tube)
  - Uterus
  - Vagina
  - Breast
  - Egg
  - Ovary
  - Cervix
- Digestive System
  - Esophagus
  - Liver
  - Mouth
  - Intestine
  - Abdomen
  - Stomach
- Male Reproductive System
  - Penis
  - Prostate gland
  - Testis
- Urinary System
  - Bladder
  - Ureters
  - Urethra
  - Kidney
- Musculoskeletal System
  - Synovial tissue
  - Muscle
  - Tendo
  - Bone

## Citation

The `cell type RAG` files were generated using data from:

**CellMarker: a manually curated resource of cell markers in human and mouse**  
*Published in Nucleic Acids Research, 2018*  
DOI: [10.1093/nar/gky900](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gky900/5115823)



