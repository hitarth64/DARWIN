import numpy as np
import pandas as pd
import matminer.featurizers.composition as comp_feat
from matminer.featurizers.base import MultipleFeaturizer
from scipy import stats
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.preprocessing import LabelEncoder
from pymatgen.core.periodic_table import Element
from matminer.utils.data import MagpieData, PymatgenData
from pymatgen.core import Composition
from typing import List

class Interpretability():
    def __init__(self, control: List[Composition], treatment: List[Composition]) -> None:
        """control and treatment represents materials that do not meet the specifications and those that do
        two methods: [statistical] vs [randomforests]
        significance test: [Kruskal-Wallis H-test]
        new methods will be added in future releases of DARWIN"""
        self.population = {"control":control, "treatment":treatment}
        self.props = ['BandCenter', 'AtomicOrbitals', 'ValenceOrbital', \
                      'YangSolidSolution','Stoichiometry'] #'ElementProperty' , 'CohesiveEnergy'
        
        # Train encoders for categorical data
        le_orbitals = LabelEncoder()
        le_orbitals.fit(['As','Ap','Ad','Af'])
        le_orbitals.classes_ = np.array(['s','p','d','f'])
        le_elements = LabelEncoder()
        elemental_symbols = [Element.from_Z(i).symbol for i in range(1,108)]
        prefixed_elemental_symbols = []
        for idx, symbol in enumerate(elemental_symbols):
            prefixed_elemental_symbols.append(chr(97+idx//26)+chr(97+idx%26)+symbol) 
        le_elements.fit(prefixed_elemental_symbols)
        le_elements.classes_ = np.array(elemental_symbols)
        self.le_orbitals = le_orbitals
        self.le_elements = le_elements
        # generate features that will be used for analysis
        self.features = self.featurize()
        # chemical rules generated by DARWIN approach
        self.rules = {}
        
    def population_sizes(self, pop_type: str) -> int:
        """pop_type: control OR treatment
        returns population sizes of the two groups"""
        return len(self.population[pop_type])
    
    def analysis(self, method:str='statistical') -> None:
        """performs the analysis of choice and returns chemical design rules alongwith their statistical significance
        method choices: 'statistical' OR 'randomforest' to calculate feature importances
        statistical significance: returned through Kruskal-Wallis H-test"""
        if method == 'statistical':
            self.rules = self.spearman_analysis(self.features)
        elif method == 'randomforest':
            self.rules = self.foresting(self.features)
        
    def significance_analysis(self, alpha:float=0.05) -> None:
        """Use Kruskal-Wallis H-test to identify statistically significant rules
        H-test is preferred over ANOVA due to absence of normal distribution assumption
        Focus on statistical significance"""
        for col in self.features['control'].columns.values:
            try:
                val = stats.kruskal(self.features['control'][col], self.features['treatment'][col])
                self.rules[col] = {'importance':self.rules[col], 'pvalue':val.pvalue}
            except:
                #print("Problem with column: {}".format(col))
                self.rules[col] = {'importance':self.rules[col], 'pvalue':1.0}

    def spearman_analysis(self, features: pd.DataFrame, alpha:float=0.05) -> dict:
        """Use Spearman-R to identify chemical rules importance"""
        significant_rules = {}
        for col in features['control'].columns.values:
            X = pd.concat([features["control"][col], features["treatment"][col]],axis=0).values
            y = np.array([0]*features["control"].shape[0] + [1]*self.features["treatment"].shape[0])
            val = stats.spearmanr(X, y)
            significant_rules[col] = val.correlation
        return dict(sorted(significant_rules.items(), key=lambda item: item[1])) 
    
    def foresting(self, features: pd.DataFrame) -> dict:
        """Use Random forest to identify chemical rules importance"""
        # Prepare data
        X = pd.concat([features["control"], features["treatment"]],axis=0).values
        y = np.array([0]*features["control"].shape[0] + [1]*self.features["treatment"].shape[0])
        
        reg = RandomForestClassifier(n_estimators=100, random_state=0)
        # Perform cross-validation to identify the best set of hyper-parameters
        space  = [Integer(10, 500, name='max_depth'),
          Categorical([True, False],name='bootstrap'),
          Categorical([None, 'log2', 'sqrt'], name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]
        
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -1*np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=1))
        
        res_gp = gp_minimize(objective, space, n_calls=40, random_state=0)
        
        if -1*res_gp.fun < 0.7:
            print('Current CV accuracy (',int(-1*res_gp.fun*100) ,'%)is less than 70%. Please be careful about its usage.')
            print('Maybe try other interpretability methods for the population under consideration')
            
        optimal_params = dict(zip(res_gp.space.dimension_names, res_gp.x))
        reg = RandomForestClassifier(**optimal_params)
        reg = reg.fit(X,y)
        importances = dict(zip(features["control"].columns.values, reg.feature_importances_))
        return dict(sorted(importances.items(), key=lambda item: item[1])) 
    
    def sorted_diff(self, input_array: list) -> np.array:
        """
        Returns all possible sorted differences 
        input_array: list-like object
        """
        input_array.sort()
        sorted_vals = input_array[::-1]
        return np.array([sorted_vals[i]-sorted_vals[j] for i in range(0,len(input_array)) for j in range(i,len(input_array))])
        
    def ratio(self, input_data):
        """
        Add all possible distinct ratios to the input dataframe and returns it
        input_data: Pandas dataframe of [compounds X props]
        """
        props = input_data.columns.values
        new_prop_names, new_props = [], []
        for i in range(len(props)):
            for j in range(i+1,len(props)):
                if 0 not in input_data[props[j]].values:
                    new_prop_names.append(props[i]+' : '+props[j])
                    new_props.append( list( input_data[props[i]].values/input_data[props[j]].values ) )
        new_props_df = pd.DataFrame(new_props, index=new_prop_names).T
        new_props_df = pd.concat([input_data, new_props_df],axis=1)
        return new_props_df
        
    def featurize(self, properties=None, statistical_quantities=None, operations=None):
        """
        Featurizes the inputs: accepts optional operations and properties arguments
        If specified, properties has to be provided as a dictionary of source and property-name
        """
        
        if properties is None:
            features = ["Number","MendeleevNumber","AtomicWeight","MeltingT","CovalentRadius",\
                        "Electronegativity","NsValence","NpValence","NdValence","NfValence","NValence",\
                        "NsUnfilled","NpUnfilled","NdUnfilled","NfUnfilled","NUnfilled",\
                        "GSbandgap","GSmagmom","Row","Column"]
            properties = {'magpie':features}
        
        prange = lambda x: np.max(x) - np.min(x) # calculates range of property array
        if statistical_quantities is None:
            statistical_quantities = {'mu_':np.mean, 'std_':np.std, 'sum_':np.sum,\
                                      'max_':np.max, 'min_':np.min, 'range_':prange} ## 'sodi_':self.sorted_diff, \
            
        if operations is None:
            operations = [self.ratio]
            
        def comp_elem_featurize_many(source, comps, prop):
            """
            comps: list of compositions
            """
            complete_prop = {}
            for sq in statistical_quantities.items():
                complete_prop[sq[0]+source+'_'+prop] = {i:sq[1](source_mapping[source].get_elemental_properties(comps[i].elements,prop)) for i in range(len(comps))}
            return complete_prop
            
        source_mapping = {'magpie':MagpieData(), 'pmg':PymatgenData()}
        quantity = {'control':{}, 'treatment':{}}
        for source in properties.keys():
            for prop in properties[source]:
                quantity['control'].update(comp_elem_featurize_many(source, self.population['control'], prop))
                quantity['treatment'].update(comp_elem_featurize_many(source, self.population['treatment'], prop)) 
        
        quantity['control'] = pd.DataFrame(quantity['control'])
        quantity['treatment'] = pd.DataFrame(quantity['treatment'])
        output = {'control':{}, 'treatment':{}}
        
        for op in operations:
            output['control'] = self.ratio(quantity['control'])
            output['treatment'] = self.ratio(quantity['treatment'])
            
        common_columns = np.intersect1d(output['control'].columns.values, output['treatment'].columns.values)
        output['control'] = output['control'][common_columns]
        output['treatment'] = output['treatment'][common_columns]
        self.intermediate_output = output
        
        #========================
        #=====Other features=====
        #========================
        mf = MultipleFeaturizer([getattr(comp_feat, i)() for i in self.props])
        cols = mf.feature_labels()
        features_control = pd.DataFrame(mf.featurize_many(self.population['control'],ignore_errors=True), columns=cols)
        features_control = features_control.dropna(axis=0)
        features_treatment = pd.DataFrame(mf.featurize_many(self.population['treatment'],ignore_errors=True), columns=cols)
        features_treatment = features_treatment.dropna(axis=0)
        
        for col in ["HOMO_character","LUMO_character"]:
            features_control[col] = self.le_orbitals.transform(features_control[col])
            features_treatment[col] = self.le_orbitals.transform(features_treatment[col])
            
        for col in ["HOMO_element","LUMO_element"]:
            features_control[col] = self.le_elements.transform(features_control[col])
            features_treatment[col] = self.le_elements.transform(features_treatment[col])
            
        col1 = features_control.dropna(axis=1).columns.values
        col2 = features_treatment.dropna(axis=1).columns.values
        col = np.intersect1d(col1,col2)
            
        output['control'] = pd.concat([output['control'].loc[features_control.index.values], features_control[col]], axis=1)
        output['treatment'] = pd.concat([output['treatment'].loc[features_treatment.index.values], features_treatment[col]], axis=1)
        return output