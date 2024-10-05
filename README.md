# scChat: A Large Language Model-Powered Co-Pilot for Contextualized Single-Cell RNA Sequencing Analysis
Welcome to the [scChat paper](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) page. scChat is a pioneering AI assistant designed to enhance single-cell RNA sequencing (scRNA-seq) analysis by incorporating research context into the workflow. Powered by a large language model (LLM), scChat goes beyond standard tasks like cell annotation by offering advanced capabilities such as research context-based experimental analysis, hypothesis validation, and suggestions for future experiments.

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
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Chat Example](#chat-example)
- [Citation](#citation)

# Overview
<a name="overview"></a>
## Motivation
Data-driven methods such as unsupervised and supervised learning are essential tools in single-cell RNA sequencing (scRNA-seq) analysis. However, these methods often lack the ability to incorporate research context, which can lead to missed insights. scChat addresses this by integrating contextualized conversation with data analysis to provide a deeper understanding of experimental results. It supports the exploration of research hypotheses and generates actionable insights for future experiments.

Please read our [scChat paper](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) for more motivation and details about how the scChat works.

## Scope
Model: scChat currently supports analysis using AnnData-formatted single-cell RNA sequencing datasets.

Capabilities: scChat integrates an LLM with specialized tools to enable tasks such as marker gene identification, UMAP clustering, and custom literature searches, all through conversational interactions.

Future Work: Future versions will include enhanced experimental design features and additional validation mechanisms to further refine scRNA-seq analysis.

# Installation

To set up the project environment and run the server, follow these steps:

1. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt

Follow these steps to utilize the application effectively:
### Step 1: Initialize the Application
- run python3 manage.py runserver

### Step 2: Access the Application
- Go to http://127.0.0.1:8000/schatbot on a web browser.



# Tutorial
<a name="tutorial"></a>
1. Upload adata file
2. Upload sample mapping (.json file) (if required).
3. request to generate UMAP for RNA Analysis.
4. (3) Will return a python dictionary, type in that you want to label/annotate clusters for overall cells. 
5. Now you can ask to display annotated umap for overall cells or view the non-annotated umap for overall cells.
6. You can ask for rationale or research questions specific to your dataset.
7. If you want you can filter and process a specific cell type.
8. (7) would return a python dictionary, type in that you want to label/annotate clusters for the processed cell type
9. You can ask for reasoning, possible hypothesis and so on.

## Chat Example
<a name="chat-example"></a>
<p align="center">
<!-- <img src="images/Chatbot_eg_highPPI.png" alt="drawing" width="700"/> -->
</p>



# Model Library:



# Citation
<a name="citation"></a>
Cite us
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


