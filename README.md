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
- [Datasets](#datasets)

# Overview
<a name="overview"></a>
## 1. Motivation
Data-driven methods such as unsupervised and supervised learning are essential tools in single-cell RNA sequencing (scRNA-seq) analysis. However, these methods often lack the ability to incorporate research context, which can lead to overlooked insights. scChat addresses this by integrating contextualized conversation with data analysis to provide a deeper understanding of experimental results. It supports the exploration of research hypotheses and generates actionable insights for future experiments.

Please read our [scChat paper](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) for more motivation and details about how the scChat works.

## 2. Scope
Model: scChat currently supports analysis using AnnData-formatted single-cell RNA sequencing datasets.

Capabilities: scChat integrates an LLM mutli-agent system with specialized tools to enable tasks, such as cell type annotation, enrichment analysis, and result visualization, all through conversational interactions.

## 3. Methodology
<a name="system overview"></a>
<p align="center">
<img src="images/scchat framework.png" alt="drawing" width="500"/>
</p>
scChat – a multi-agent scRNA-seq research co-scientist – that can autonomously generate executable plans for multi-step analyses, ranging from data preprocessing and follow-up analysis to results visualization.
scChat includes five main agents in it:

<details>
<summary><strong>🧠 Planner</strong></summary>
Searches for function execution and conversation history, parses the query, and decomposes it to generate a plan with several function calls arranged as steps in sequence.
</details>

<details>
<summary><strong>⚡ Executor</strong></summary>
Performs the function specified in the plan iteratively.
</details>

<details>
<summary><strong>✅ Evaluator</strong></summary>
Validates the outcome of each function from the executor, handling errors and interrupting the plan to pass error messages to the response generator if needed. Additionally, it checks the availability of remaining steps and determines the next step in the workflow.
</details>

<details>
<summary><strong>🔍 Critic</strong></summary>
Identifies potentially missing functions by creating a separate plan based on the function results, ensuring targeted analyses of specific cell types with all necessary downstream steps.
</details>

<details>
<summary><strong>📝 Response Generator</strong></summary>
Compiles all relevant function results to generate the final response to the user's query. After generating the response, it stores the final response and the function execution results in conversation and function histories, respectively.
</details>

# Tutorial 

To set up the project environment and run the server, follow these steps:

### Step 1: Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

### Step 2: Set the OPENAI Key Environment Variable 
- Type and enter export OPENAI_API_KEY='your_openai_api_key' in your terminal

  
### Step 3: Download Neo4j Desktop 2
- Download Neo4j Desktop 2 (https://neo4j.com/download/)
- Download required dump files (https://drive.google.com/drive/folders/17UCKv95G3tFqeyce1oQAo3ss2vS7uZQE)
- Create a new instance on Neo4j (this step asks you set the password)
- Import the dump files as new databases in the created instance.
- Start the database

### Step 4: Upload and update files
- Upload scRNA-seq adata file (.h5ad)
- Upload the pathway vector-based model (.pkl and .faiss), which can be found in this link: https://drive.google.com/drive/u/4/folders/1OklM2u5T5FsjiUvvYRYyWxrssQIb84Ky
- Update specification_graph.json with your Neo4j username, password, system and organ relevant to the database you are using with specific format
- Update sample_mapping.json with adata file corresponding "Sample name", which can be found in adata.obs, and write descriptions for each condition.

### Step 5: Build the specification.json and sample_mapping.json for RAG specifications
- Build the specification_graph.json with your Neo4j username, password, database(`human` or `mouse`), system and organ relevant to the file you are going to test with following format:
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
- Build the sample_mapping.json with adata file corresponding "Sample name", which can be found in adata.obs, and write descriptions for each condition. For example:
```json
{
  "Sample name": "Sample",
  "Sample categories": {
    "0": "p1_pre",
    "1": "p1_post",
    "2": "p6_pre",
    "3": "p6_post",
    "4": "p7_pre",
    "5": "p7_post"
  },
  "Sample description": {
    "p1_pre": "Pre-treatment sample from patient 1",
    "p1_post": "Post-treatment sample from patient 1",
    "p6_pre": "Pre-treatment sample from patient 6",
    "p6_post": "Post-treatment sample from patient 6",
    "p7_pre": "Pre-treatment sample from patient 7",
    "p7_post": "Post-treatment sample from patient 7"
  }
}
```
Notably, the available systems, organs and tissues are listed in available_cell_RAG.json.

### Step 6: Initialize the Application
- Run python3 manage.py migrate (For the first time as you install scChat)
- Run python3 manage.py runserver

### Step 7: Access the Application
- Open your web browser and navigate to:
  `http://127.0.0.1:8000/schatbot`
  
- **Recommended:** Use Google Chrome for the best experience.

### Additional Tip: Clear Cache to Avoid Previous Chat Data
- Periodically clearing the cache is recommended to ensure a smooth experience:
  1. Right-click on the page and select **Inspect**.
  2. Go to the **Application** tab.
  3. Under **Cookies**, remove `sessionid`.
  4. You may have to run `python manage.py migrate` in some cases before Step 4.
  
  This will prevent previous chat sessions from being reprocessed.

## Chat Example
<a name="chat-example"></a>
<p align="center">
<!-- <img src="images/Chatbot_eg_highPPI.png" alt="drawing" width="700"/> -->
</p>

# Datasets
The datasets used for testing and examples for sample_mapping.json and specification_graph.json can be found at [https://docs.google.com/spreadsheets/d/1NwN5GydHn0B3-W0DLcAfvnNtZVJEMUgBW9YyzXnS83A/edit?usp=sharing
](https://drive.google.com/drive/u/4/folders/1RJRETtwI3zxsOJK0Lop197JGm3Isl4iB)