import torch
from matdeeplearn import models
from matdeeplearn import models, process

# choose device from cpu, mps and cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Config on graph generation from crystal
graph_processing = {'SM_descriptor': 'False',
                'SOAP_descriptor': 'False',
                'SOAP_lmax': 4,
                'SOAP_nmax': 6,
                'SOAP_rcut': 8.0,
                'SOAP_sigma': 0.3,
                'data_format': 'cif',
                'data_path': 'dummy_data',
                'dataset_type': 'inmemory',
                'dictionary_path': 'atom_dict.json',
                'dictionary_source': 'default',
                'edge_features': 'True',
                'graph_edge_length': 50,
                'graph_max_neighbors': 12,
                'graph_max_radius': 8.0,
                'target_path': 'targets.csv',
                'verbose': 'True',
                'voronoi': 'False'}


dummy_dataset = process.get_dataset('darwin/models/dummy_data', 0, False, graph_processing)

def return_base_model():
    """Return a base model for the three properties."""
    model = models.MEGNet(dummy_dataset, dim1=190, dim2=180, dim3=90, gnn_count=6,\
        lr=0.0080614, pool='set2set', post_fc_count=5, pre_fc_count=1, gc_count=4, gc_fc_count=1)
    return model

def return_ehull_bandgap_din_models():
    # Initialize and load saved weights

    ehull_model = return_base_model()
    ehull_weights = torch.load('darwin/models/models_used/ehull_model.pth', map_location=device)
    ehull_model.load_state_dict(ehull_weights)

    bandgap_model = return_base_model()
    bandgap_weights = torch.load('darwin/models/models_used/bandgap_model.pth', map_location=device)
    bandgap_model.load_state_dict(bandgap_weights)

    din_model = return_base_model()
    din_weights = torch.load('darwin/models/models_used/din_model_2.pth', map_location=device)
    din_model.load_state_dict(din_weights)

    return ehull_model, bandgap_model, din_model
