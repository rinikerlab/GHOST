# GHOST

This repository is part of the Supporting Information to

## GHOST: Adjusting the Decision Threshold to Handle Imbalanced Data in Machine Learning
Carmen Esposito,<sup>1</sup> Gregory A. Landrum,<sup>1,2</sup> Nadine Schneider,<sup>3</sup> Nikolaus Stiefl,<sup>3</sup> and Sereina Riniker<sup>1</sup>

<sup>1</sup> Laboratory of Physical Chemistry, ETH Zurich, Vladimir-Prelog-Weg 2, 8093 Zurich,Switzerland <br />
<sup>2</sup> T5 Informatics GmbH, Spalenring 11, 4055 Basel, Switzerland <br />
<sup>3</sup> Novartis Institutes for BioMedical Research, Novartis Pharma AG, Novartis Campus,4002 Basel, Switzerland <br />

## Content

### Notebooks:

- **example_oob_threshold_optimization.ipynb** <br />
  Notebook containing the oob-based threshold optimization function and a simple example 

- **example_GHOST.ipynb** <br />
  Notebook containing the GHOST (**G**eneralized t**H**resh**O**ld **S**hif**T**ing) function and a simple example 
  
- **Tutorial_Threshold_Optimization_RF.ipynb** <br />
  Notebook explaining step by step how to reproduce the results reported in our work.
  Here, the code is only executed for 6 public datasets and the random forest model.
  
- **Reproduce_Results_Public_Datasets.ipynb** <br />
  Notebook to reproduce the results reported in our work.
  Here, results are produced for all 138 public datasets. The user can choose between four different machine learning methods, namely random forest (RF), gradient boosting (GB), XGBoost (XGB), and logistic regression (LR). The user can also choose between two different molecular descriptors, ECFP4 and RDKit2D.
  
### Data:
The threshold optimization methods have been validated agaist 138 public datasets and these are all provided here in the folder `data`.

### Dependencies:
A list of dependencies is available in the file `ghost_env.yml`. This conda environmet was used to obtain the results reported in our work.

## Authors
[Carmen Esposito]() (GHOST procedure) and [Greg Landrum](https://github.com/greglandrum) (oob-based threshold optimization approach, data collection, [initial code](https://github.com/greglandrum/rdkit_blog/blob/master/notebooks/Working%20with%20unbalanced%20data%20part%201.ipynb)).

## Acknowledgements
Conformal prediction (CP) experiments were adapted from the [CP functions](https://github.com/volkamerlab/knowtox_manuscript_SI) provided by the [Volkamer Lab](https://volkamerlab.org/).  

## License

## Citation



