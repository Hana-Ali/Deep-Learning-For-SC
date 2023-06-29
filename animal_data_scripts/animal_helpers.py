import numpy as np

# Function that creates a dictionary of edge attributes depending on file name
def define_edge_attributes(file_name):
    # Define the main attributes depending on name
    if file_name == "drosophila_medulla_1":
        # edge_attributes = ["post.z", "post.x", "post.y", "pre.x", "pre.y", "pre.z", "proofreading.details"]
        edge_attributes = ["post.z", "post.x", "post.y", "pre.x", "pre.y", "pre.z"]
    elif file_name == "rhesus_brain_2":
        edge_attributes = ["SLN_perc", "Dist_mm"]
    elif file_name == "rhesus_cerebral_cortex_1":
        # edge_attributes = ["CASE", "STATUS", "BIBLIOGRAPHY", "MONKEY", "NEURONS", "FLNe"]
        edge_attributes = ["FLNe"]
    elif file_name == "rhesus_interareal_cortical_network_2":
        edge_attributes = ["weight"]
    elif file_name == "mouse_brain_1":
        edge_attributes = ["w_contra_weight", "pvalue_contra_weight", "w_ipsi_weight", "pvalue_ipsi_weight"]
    elif file_name == "mouse_retina_1":
        edge_attributes = ["y", "x", "z", "area"]
    elif file_name == "mouse_visual_cortex_1":
        edge_attributes = ["weight"]
    elif file_name == "mouse_visual_cortex_2":
        edge_attributes = ["weight"]
    elif file_name == "rattus_norvegicus_brain_1":
        edge_attributes = ["weight"]
    elif file_name == "rattus_norvegicus_brain_2":
        edge_attributes = ["weight"]
    elif file_name == "rattus_norvegicus_brain_3":
        edge_attributes = ["weight"]
    elif file_name == "c_elegans_neural_male_1":
        edge_attributes = ["electrical_weight", "chemical_weight"]
    elif file_name == "c_elegans_herm_pharynx_1":
        # edge_attributes = ["synapse_type", "weight"]
        edge_attributes = ["weight"]
    elif file_name == "p_pacificus_neural_synaptic_1":
        edge_attributes = ["synapse.ID"]
    elif file_name == "p_pacificus_neural_synaptic_2":
        edge_attributes = ["synapse.ID"]
    else:
        edge_attributes = []
    
    # Return the edge attributes
    return edge_attributes