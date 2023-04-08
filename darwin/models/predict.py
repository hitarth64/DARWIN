import numpy as np
import torch
from matdeeplearn import training
from torch_geometric.loader import DataLoader
from darwin.models.load_models import return_ehull_bandgap_din_models
from darwin.models import processing
from typing import List
from ase import Atoms

# choose device from cpu, mps and cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
#elif torch.backends.mps.is_available():
#    device = torch.device('mps')
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

ehull_model, bandgap_model, din_model = return_ehull_bandgap_din_models()
ehull_model, bandgap_model, din_model = ehull_model.to(device), bandgap_model.to(device), din_model.to(device)
ehull_model, bandgap_model, din_model = ehull_model.eval(), bandgap_model.eval(), din_model.eval()

def predict_properties(list_of_ase_objects:List[Atoms], ehull:bool=False, bandgap:bool=False, din:bool=False):
    """Predict properties for given list of ASE crystals."""
    
    data_, slices_, list_ = processing.process_ase_objects(list_of_ase_objects, graph_processing, edge_feature_min=0, edge_feature_max=8)
    loader = DataLoader(list_, batch_size=2048, shuffle=False)
    targets = {}
    
    if ehull:
        _, ehull_out = training.evaluate(loader, ehull_model, 'l1_loss', device, out=True)
        targets['ehull'] = ehull_out[:,-1].astype('float')
    
    if din:
        _, din_out = training.evaluate(loader, din_model, 'binary_cross_entropy_with_logits', device, out=True)
        din_out = din_out[:,-1].astype('float')
        targets['din'] = 1/(1+np.exp(-1*din_out))

    if bandgap:
        data_, slices_, list_ = processing.process_ase_objects(list_of_ase_objects, graph_processing, edge_feature_min=0, edge_feature_max=7.9956)
        loader = DataLoader(list_, batch_size=2048, shuffle=False)
        _, bandgap_out = training.evaluate(loader, bandgap_model, 'l1_loss', device, out=True)
        targets['bandgap'] = bandgap_out[:,-1].astype('float')
        
    return targets
