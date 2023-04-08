import os
import torch
import numpy as np
from matdeeplearn import process
import torch_geometric.utils as pyg_utils
from pymatgen.io.ase import AseAtomsAdaptor
from matdeeplearn.process import StructureDataset
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset

class StructureDatasetWithoutWriting(process.StructureDataset):
    def __init__(self, data, slices, data_path=None, processed_path="processed", transform=None, pre_transform=None):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = data, slices

        
def process_ase_objects(atoms_objects, processing_args, edge_feature_min, edge_feature_max):

    ##Load dictionary
    atom_dictionary = process.get_dictionary(
        os.path.join(
            './matdeeplearn/process/'
            "dictionary_default.json",
        )
    )

    ##Load targets
    target_data = [[idx,idx] for idx,st in enumerate(atoms_objects)]

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(atoms_objects)):

        structure_id = target_data[index][0]
        data = process.Data()
        data.ase = atoms_objects[index]
        ase_crystal = atoms_objects[index]
        
        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        ##Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = process.threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = pyg_utils.dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        # Adding self-loops
        edge_index, edge_weight = pyg_utils.add_self_loops(
            edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
        )
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        distance_matrix_mask = (
            distance_matrix_trimmed.fill_diagonal_(1) != 0
        ).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        target = target_data[index][1:]
        y = torch.Tensor(np.array([target], dtype=np.float32))
        data.y = y

        # pos = torch.Tensor(ase_crystal.get_positions())
        # data.pos = pos
        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data.structure_id = [[structure_id] * len(data.y)]
        data_list.append(data)

    ##
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)

    ##Generate node features
    for index in range(0, len(data_list)):
        atom_fea = np.vstack(
            [
                atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))
            ]
        ).astype(float)
        data_list[index].x = torch.Tensor(atom_fea)

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = process.OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    ##Generate edge features
    ##Distance descriptor using a Gaussian basis
    distance_gaussian = process.GaussianSmearing(
        0, 1, processing_args["graph_edge_length"], 0.2
    )
    # print(GetRanges(data_list, 'distance'))
    process.NormalizeEdge(data_list, "distance", edge_feature_min, edge_feature_max)
    # print(GetRanges(data_list, 'distance'))
    for index in range(0, len(data_list)):
        data_list[index].edge_attr = distance_gaussian(
            data_list[index].edge_descriptor["distance"]
        )

    process.Cleanup(data_list, ["ase", "edge_descriptor"])

    data, slices = InMemoryDataset.collate(data_list)
    return data, slices, data_list