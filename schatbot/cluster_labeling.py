import ast
import pandas as pd
import pickle
import os

# Globals used by these functions (make sure these are set appropriately elsewhere)
adata = None
annotated_adata = None
conversation_history2 = None
current_adata = None
base_annotated_adata = None

def label_clusters(cell_type: str):
    """
    Labels clusters using a mapping extracted from the conversation history.
    Saves annotated UMAP data and returns a confirmation string.
    """
    global adata, annotated_adata, conversation_history2, current_adata, base_annotated_adata
    adata2 = adata.copy()
    standardized_cell_type3 = cell_type.split()[0].capitalize()
    standardized_cell_type2 = cell_type.split()[0].capitalize() + " cell"
    standardized_cell_type = cell_type.split()[0].capitalize() + " cells"
    last_message = conversation_history2[-2]['content']
    try:
        start_idx = last_message.find("{")
        end_idx = last_message.rfind("}") + 1
        str_map = last_message[start_idx:end_idx]
        map2 = ast.literal_eval(str_map)
        map2 = {str(key): value for key, value in map2.items()}
        if standardized_cell_type == "Overall cells" or standardized_cell_type2 == "Overall cell":
            adata2.obs['cell_type'] = 'Unknown'
            for group, cl_type in map2.items():
                adata2.obs.loc[adata2.obs['leiden'] == group, 'cell_type'] = cl_type
            adata2.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']]\
                  .to_csv(f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            annotated_adata = adata2.copy()
            fname = f'annotated_adata/Overall cells_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)
            base_annotated_adata = adata2
        else:
            adata3 = base_annotated_adata.copy()
            specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type2])].copy()
            if specific_cells.shape[0] == 0:
                specific_cells = adata3[adata3.obs['cell_type'].isin([standardized_cell_type3])].copy()
            import scanpy as sc
            sc.tl.pca(specific_cells, svd_solver='arpack')
            sc.pp.neighbors(specific_cells)
            sc.tl.umap(specific_cells)
            sc.tl.leiden(specific_cells, resolution=0.5)
            specific_cells.obs['cell_type'] = 'Unknown'
            for group, cl_type in map2.items():
                specific_cells.obs.loc[specific_cells.obs['leiden'] == group, 'cell_type'] = cl_type
            specific_cells.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']]\
                          .to_csv(f'umaps/{standardized_cell_type}_annotated_umap_data.csv', index=False)
            annotated_adata = specific_cells.copy()
            fname = f'annotated_adata/{standardized_cell_type}_annotated_adata.pkl'
            with open(fname, "wb") as file:
                pickle.dump(annotated_adata, file)
    except (SyntaxError, ValueError) as e:
        return f"Error in parsing the map: {e}"
    return "Repeat 'Annotation of clusters is complete'"

def convert_into_labels(conversation_history):
    """
    Extracts a mapping from the last conversation message and updates cell labels.
    Saves annotated data to CSV and a pickle file.
    """
    global adata, annotated_adata
    last_message = conversation_history[-1]['content']
    str_map = ""
    flag = False
    for char in last_message:
        if char == "{":
            flag = True
        if flag:
            str_map += char
        if char == "}":
            flag = False
    map2 = ast.literal_eval(str_map)
    map2 = {str(key): value for key, value in map2.items()}
    adata.obs['cell_type'] = 'Unknown'
    for group, cl_type in map2.items():
        adata.obs.loc[adata.obs['leiden'] == group, 'cell_type'] = cl_type
    adata.obs[['UMAP_1', 'UMAP_2', 'leiden', 'patient_name', 'cell_type']]\
         .to_csv("annotated_umap_data.csv", index=False)
    annotated_adata = adata.copy()
    with open('overall_annotated_adata.pkl', 'wb') as file:
        pickle.dump(annotated_adata, file)