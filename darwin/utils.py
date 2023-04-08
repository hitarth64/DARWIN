from pymatgen.core import Structure
from typing import List, Tuple
from pymatgen.core import Element
from mp_api.client import MPRester

def get_pblock_compounds() -> Tuple[List[Structure], List[int]]:
    # returns compounds from materials project with only elements from the p-block
    excluded_elements = []
    for z in range(1, 84):
        elm = Element.from_Z(z)
        if elm.group not in [13, 14, 15, 16, 17]:
            excluded_elements.append(elm.symbol)
    mpr = MPRester('qM3y7u1PS9P4uhKOTZiiLEsHBfL0wgNK')
    queries = mpr.summary.search(num_sites=(1,20), num_elements=2, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures = [q.initial_structures[-1].get_primitive_structure() for q in queries]
    return structures, [Element(e).Z for e in excluded_elements]
    

def get_halide_based_compounds() -> Tuple[List[Structure], List[int]]:
    # atomic numbers for nobel gases, Tc, Cd, Pb, Hg, Tl, As and expensive elements such as Au, Pt, Ir, Pd, Rh, Ru
    exluded_atomic_numbers = [2, 10, 18, 36, 54, 86, 43, 44, 45, 46, 77, 78, 79, 80, 81, 82, 33, 34, 52]
    excluded_elements = [Element.from_Z(z).symbol for z in exluded_atomic_numbers]
    mpr = MPRester('qM3y7u1PS9P4uhKOTZiiLEsHBfL0wgNK')
    queries = mpr.summary.search(elements=['Cl'], num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures = [q.initial_structures[-1].get_primitive_structure() for q in queries]
    queries = mpr.summary.search(elements=['Br'], num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures += [q.initial_structures[-1].get_primitive_structure() for q in queries]
    queries = mpr.summary.search(elements=['I'], num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures += [q.initial_structures[-1].get_primitive_structure() for q in queries]
    return structures, [Element(e).Z for e in excluded_elements]


def get_ABX3_compounds() -> Tuple[List[Structure], List[int]]:
    # atomic numbers for nobel gases, Tc, Cd, Pb, Hg, Tl, As and expensive elements such as Au, Pt, Ir, Pd, Rh, Ru
    exluded_atomic_numbers = [2, 10, 18, 36, 54, 86, 43, 44, 45, 46, 77, 78, 79, 80, 81, 82, 33, 34, 52]
    excluded_elements = [Element.from_Z(z).symbol for z in exluded_atomic_numbers]
    mpr = MPRester('qM3y7u1PS9P4uhKOTZiiLEsHBfL0wgNK')
    queries = mpr.summary.search(formula='**Cl3', num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures = [q.initial_structures[-1].get_primitive_structure() for q in queries]
    queries = mpr.summary.search(formula='**Br3', num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures += [q.initial_structures[-1].get_primitive_structure() for q in queries]
    queries = mpr.summary.search(formula='**I3', num_sites=(1,20), num_elements=3, fields=['material_id'], exclude_elements=excluded_elements)
    queries = mpr.materials.search(material_ids=[r.material_id for r in queries], fields=['initial_structures'])
    structures += [q.initial_structures[-1].get_primitive_structure() for q in queries]
    return structures, [Element(e).Z for e in excluded_elements]


class GASearchParams():
    """Parameters for the genetic algorithm search"""
    def __init__(self, properties:List[str]=['ehull', 'bandgap', 'din'], targets:List[float]=[0.03,1.5,1.0]):
        
        self.attempts = 2
        self.population_size = 10
        self.generations = 10
        self.mutation_rate = 0.1
        self.crossover = True
        self.keep_n = 0.4
        
        self.relevant_properties = {'ehull':False, 'bandgap':False, 'din':False}
        self.properties = properties
        self.targets = targets
        self.weights = [1/len(properties)]*len(properties)

        for property in self.properties:
            self.relevant_properties[property] = True

        assert len(self.properties) == len(self.targets) ,\
                     "The number of properties, and targets must be the same"