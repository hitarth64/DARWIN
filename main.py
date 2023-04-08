from darwin.optimize import search_and_optimize
from darwin.utils import GASearchParams, get_halide_based_compounds, get_ABX3_compounds, get_pblock_compounds
from pymatgen.io.ase import AseAtomsAdaptor as AAA
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--output', type=str, default='interpretability_compositions.pkl')
parser.add_argument('--generational', type=str, default='generational_outcomes.pkl')
args = parser.parse_args()

params = GASearchParams(['ehull', 'din'], [0.03, 1.0])
params.population_size = 50
params.generations = 50
params.attempts = 100
sizes = args.size

population, excluded_elements = get_pblock_compounds() # change this to the choice of initial population you want

generational_records = search_and_optimize(params, population, excluded_elements=excluded_elements)
with open(args.generational,'wb') as f:
    pickle.dump(generational_records, f)


control_comps, treatment_comps = {}, {}

for attempt in generational_records:
    for generation in generational_records[attempt]:        
        structures = generational_records[attempt][generation]['population'].copy()
        scores = generational_records[attempt][generation]['badness'].copy()
        for st,score in zip(structures, scores):
            if score > 0.5:
                control_comps[st.composition] = score
            elif score < 0.3:
                treatment_comps[st.composition] = score
        
control_comps = [k for k, v in sorted(control_comps.items(), key=lambda item: item[1], reverse=True)][:sizes]
treatment_comps = [k for k, v in sorted(treatment_comps.items(), key=lambda item: item[1])][:sizes]
print('Size of control set:', len(control_comps))
print('Size of treatment set: ', len(treatment_comps))
control_comps = list(control_comps)
treatment_comps = list(treatment_comps)
with open(args.output, 'wb') as f:
    pickle.dump([control_comps, treatment_comps], f)