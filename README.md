# Deep Adaptive Regressive Weighted Intelligent Network (DARWIN)

This Repository implements the DARWIN approach for interpretable discovery of materials.
In the process, the package provides following functionalities:

* Enables search for promising materials with specified target properties using evolutionary algorithm
* Enables interpretability analysis on the predictions of the machine learning model and evolutionary algorithm
* Provides trained and ready-to-use machine learning models for energy-above-hull, bandgaps and direct-indirect nature of the bandgap
* Transfer learning of pre-trained models
* The following paper describes the details of the DARWIN framework: [DARWIN](https://www.nature.com/articles/s41524-023-01066-9)

## Table of Contents:
- [Requirements](#requirements)
- [Usage](#usage)
  - [Using existing prediction](#using-existing-predictions)
  - [Using EA with pretrained models](#using-EA-with-pretrained-models)
  - [Interfacing DARWIN with custom models](#interfacing-DARWIN-with-custom-models)
  - [Transfer learning](#transfer-learning)
- [Data](#data)
- [Authors](#authors)
- [License](#license)
- [How to cite](#how-to-cite) 


## Requirements:
DARWIN shares all the dependencies with [MatDeepLearn](https://github.com/vxfung/MatDeepLearn) that was forked and used for performing hyper-parameter optimizations for different possible neural networks. Therefore, please follow the following steps to use DARWIN:
* Clone this repository
* Install all the [MatDeepLearn requirements](https://github.com/vxfung/MatDeepLearn#prerequisites)
* We provide a fork of MatDeepLearn to enable predictions on-the-fly for smaller sets of materials
* Once all the requirements are installed, you should add DARWIN to your system path as:
```
import sys
sys.path.append('location-of-DARWIN')
```

## Data: 
* All the data used in this study was obtained from OQMD (for pre-training the model), Materials Project (for fine-tuning on energy-above-hull and indirect-direct nature of bandgap) and [SUNMAT](https://www.snumat.com/) (for fine-tuning on HSE06 bandgaps)

## Usage:

### Using EA with models of this study:
* You can use our pre-trained models to search for desirable targets and generate insights from the run
* Please use the ```python main.py --output interpretability_groups.pkl --generational generational_data.pkl``` for performing the analysis. 
* You can specify the set or subset of properties that are of interest in the ```main``` function and their corresponding ranges.
* The generated candidates will be stored at the ```output``` filename provided above.
* All the candidates, their corresponding properties and fitness (we use the term "badness" which means negative of the fitness) values are stored in the ```generational``` filename.
* Use [interpretability.ipynb](interpretability.ipynb) for running and visualizing the analysis.

### Using existing predictions:
* It is possible to use DARWIN with one's pre-existing predictions. 
* The predictions should be provided in two columns: **control** and **treatment**.
* You can run the DARWIN analysis using the notebook provided in the repository.
* Methods available as of now for analysis are: **spearman** (Kruskal-Wallis test and spearman correlation) and **Random Forest permutation imporatance**.
* The notebook also provides a clean way to represent the findings in a histogram formatted with latex.

### Interfacing DARWIN with custom models:
* Yet another way to run DARWIN is on models not provided by us
* Users can choose the modules they want to use: EA or unsupervised analysis or both

### Transfer learning:
* You can find the support to perform transfer learning using the [TL](TL) module.
* We provide checkpoints for two pre-trained models that have been trained to predict OQMD formation energies from the relaxed structures.

## Authors
This software was written by Hitarth Choubisa.

## License
DARWIN is released under the [MIT License](LICENSE).

## How to cite:
Please cite the following work if you use DARWIN:
```
@article{Choubisa2023,
author={Choubisa, Hitarth
and Todorovi{\'{c}}, Petar
and Pina, Joao M.
and Parmar, Darshan H.
and Li, Ziliang
and Voznyy, Oleksandr
and Tamblyn, Isaac
and Sargent, Edward H.},
title={Interpretable discovery of semiconductors with machine learning},
journal={npj Computational Materials},
year={2023},
month={Jun},
day={29},
volume={9},
number={1},
pages={117},
issn={2057-3960},
doi={10.1038/s41524-023-01066-9},
url={https://doi.org/10.1038/s41524-023-01066-9}
}


```
