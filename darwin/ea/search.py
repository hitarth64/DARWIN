# Module class that creates new crystal structures using mutations and crossovers
# This module should accept a list of initial structures and generate a new list of structures
# Use Pymatgen to apply mutations and crossovers

import random
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.periodic_table import Element
from typing import List

def get_oxidation_states_records(block:list=['s','p','d','f'], max_Z:int=84, 
                                    min_row:int=1, max_row:int=6, min_group:int=1, 
                                    max_group:int=18, excluded_elements:list=[]) -> dict:
    """Get a dictionary of oxidation states and the elements that have that oxidation state.
    Rows and groups are closed intervals i.e., [min_row, max_row] and [min_group, max_group]"""

    oxidation_state_records = {}
    for i in range(1, max_Z):
        if i in [2,10,18,36,54] or i in excluded_elements:
            continue
        e = Element.from_Z(i)
        if e.block not in block and e.row not in range(min_row, max_row+1) and e.group not in range(min_group, max_group+1):
            continue
        for oxi in e.oxidation_states:
            if oxi not in oxidation_state_records:
                oxidation_state_records[oxi] = [e.symbol] 
            else:
                oxidation_state_records[oxi] += [e.symbol]
    return oxidation_state_records


def mutate(structure: Structure, oxidation_state_records:dict, seed: int = None) -> Structure:
    """Mutate a structure using ASE and Pymatgen by mutating one element with another element"""
    # Use oxidation state to determine the replacement element

    composition = structure.composition
    elms = [e.symbol for e in composition.elements]

    if seed:
        random.seed(seed)
    replacement_elm = random.choice(elms)

    try:
        oxidation_states = composition.oxi_state_guesses()
    except ValueError:
        return None
    
    if len(oxidation_states) < 1:
        return None

    oxidation_states = oxidation_states[0]
    replacement_oxi = oxidation_states[replacement_elm]

    if seed:
        random.seed(seed)

    tries = 0

    while tries < 10:
        
        if replacement_oxi % 1 != 0 or replacement_oxi not in oxidation_state_records.keys():
            return None

        new_element = random.choice(oxidation_state_records[replacement_oxi])
        if new_element == replacement_elm:
            new_element = random.choice(oxidation_state_records[replacement_oxi])
        
        # Use Pymatgen to mutate the structure
        mutated_structure = structure.copy()
        mutated_structure.replace_species({replacement_elm: new_element})

        if len(mutated_structure.composition.oxi_state_guesses()) > 0:
            return mutated_structure

        else:
            tries += 1
    
    return None


def crossover(structure1: Structure, structure2: Structure) -> Structure:
    # if two structures have same anonymous formula, then elements are transferred from one to the other
    # This first checks if the structures have the same anonymous formula and if the prototypes are different

    comp1 = structure1.composition
    comp2 = structure2.composition

    formula1 = comp1.anonymized_formula
    formula2 = comp2.anonymized_formula

    Matcher = StructureMatcher()
    same_structure = Matcher.fit(structure1, structure2)

    if formula1 == formula2 and not(same_structure):
        transfer_from = structure1.copy()
        transfer_to = structure2.copy()
        
        # Map elements from one structure to the other
        map_elements = {}
        dict1 = transfer_to.composition.as_dict()
        dict2 = transfer_from.composition.as_dict()

        for e1 in dict1.keys():
            for e2 in dict2.keys():
                if dict1[e1] == dict2[e2]:
                    map_elements[e1] = e2
        
        transfer_to.replace_species(map_elements)
        return transfer_to

class Search:

    def __init__(self, mutation_rate:float=0.5, crossover:bool=False, tries:int = 50) -> None:

        assert mutation_rate < 1, "Mutation rate should be <=1"
        self.mutation_rate = mutation_rate
        self.crossover = crossover
        self.max_tries = tries

    def generate_new_structures(self, structures: List[Structure], oxidation_state_records: dict,  keep_n: float = 0.4) -> List[Structure]:
        """Generate new structures using mutations and crossovers"""

        keep_n_structures = int(len(structures) * keep_n)
        new_structures = structures[:keep_n_structures]
        testing_structures = structures[keep_n_structures:]

        tries = 0

        while len(new_structures) < len(structures):

            st1, st2 = random.sample(testing_structures, 2)

            if self.crossover:
                st = crossover(st1, st2)
                if st is not None and len(new_structures) < len(structures):
                    new_structures.append(st)

            if st is None and len(new_structures) < len(structures):
                st = mutate(st1, oxidation_state_records)
                if st is not None:
                    new_structures.append(st)
                else:
                    tries += 1                   
                    
            if tries > self.max_tries:
                break
        
        new_structures = new_structures + random.sample(testing_structures, len(structures)-len(new_structures))
            
        return new_structures
