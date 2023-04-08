import random
import numpy as np
from darwin.models.predict import predict_properties
from pymatgen.core import Structure
from typing import List, Union, Dict
from darwin.utils import GASearchParams
from darwin.ea.search import Search, get_oxidation_states_records
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.core import Structure

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


def search_and_optimize(params:GASearchParams, original_population:List[Structure], excluded_elements=[]) -> Dict[int, Dict[int, Dict[str, Union[List[float], List[Structure]]]]]:

    search = Search(mutation_rate=params.mutation_rate, crossover=params.crossover)
    oxidation_state_records = get_oxidation_states_records(excluded_elements=excluded_elements)

    attempt = 0
    generational_records = {}

    while attempt < params.attempts:
        print(f"Attempt {attempt}")

        random.shuffle(original_population)
        population = [p for p in original_population if len(p)<20][:params.population_size]

        generational_records[attempt] = {}
        generation = 0

        while generation < params.generations:
            print(f"Generation {generation}")

            generational_records[attempt][generation] = {}
            current_record = {}

            atoms_objects_list = [AAA.get_atoms(a) for a in population]
            predictions = predict_properties(atoms_objects_list, **params.relevant_properties)                                  

            badness = np.zeros(params.population_size, dtype=float)
            for target, weight, property_name in zip(params.targets, params.weights, params.properties):
                prediction = predictions[property_name]
                current_record[property_name] = prediction
                if property_name == 'ehull':
                    badness += (prediction > target).astype("float") * weight
                elif property_name == 'bandgap':
                    badness += np.abs(prediction - target) * weight
                elif property_name == 'din' and target == 1.0:
                    badness += (prediction < 0.5).astype("float") * weight
                elif property_name == 'din' and target == 0.0:
                    badness += (prediction > 0.5).astype("float") * weight

            # Sort population by predicted properties
            population = [x for _,x in sorted(zip(badness, population), key=lambda pair: pair[0])]
            sorted_badness_indices = np.argsort(badness)
            badness = badness[sorted_badness_indices]

            for property_name in params.properties:
                generational_records[attempt][generation][property_name] = \
                    current_record[property_name][sorted_badness_indices]

            generational_records[attempt][generation]['population'] = population
            generational_records[attempt][generation]['badness'] = badness
            print(generational_records[attempt][generation]['badness'][:4], [s.composition for s in population[:4]])
            population = search.generate_new_structures(population, oxidation_state_records, keep_n=params.keep_n)

            generation += 1

        attempt += 1

    return generational_records
