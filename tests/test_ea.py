from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from ase import Atoms
from darwin.ea.search import crossover, mutate, Search
from typing import List
import pandas as pd


def test_crossover():
    x = bulk('Ni').__mul__([2,2,2])
    y = bulk('Cu').__mul__([2,2,2])
    y.set_cell([[0.0, 3.61, 3.61], [3.61, 0.0, 3.61], [3.61, 3.61, 1.0]])
    x = AseAtomsAdaptor.get_structure(x)
    y = AseAtomsAdaptor.get_structure(y)
    st = crossover(x,y)
    assert isinstance(st, Structure)


def test_mutation():
    nacl = Atoms('NaCl', [[0,1,0],[0,-1,0]], cell=[3,3,3])
    struct = AseAtomsAdaptor.get_structure(nacl)
    new_structure = mutate(struct)
    assert isinstance(new_structure, Structure)


def test_searchmethod():
    structures = pd.read_pickle('artifacts/data.pkl')
    searchmethod = Search()
    new_structures = searchmethod.generate_new_structures(structures[0:20])
    assert isinstance(new_structures, List[Structure])

if __name__ == '__main__':
    test_crossover()
    test_mutation()
    test_searchmethod()
