# Identifying AF phenotypes using a tree-based dimensionality reduction method

## Directory structure 

The provided code is to run the DDRTree algorithm, apply hierarchical clustering to form representative branches and to project other populations (in this case, the UK Biobank cohort) onto the tree. 

```
├── AF_DDRTree/                            # Contains scripts for the main analysis 
│   ├── creating_ddrtree.R                              # Main script to run the DDRTree algorithm and create the trajectory
│   ├── merging_subbranches_hier.R                      # Script to apply hierarchical clustering to merge sub-branches
│   ├── projecting_ukb_ddrtree.ipynb                    # Jupyter Notebook with code to predict tree variables (dimension coordinates and phenogroup assignments) for new dataset
│   ├── xgboost_gridsearch.py                           # Script to run the grid search for the XGBoost models predicting dimension coordinates 1 and 2, best parameters used in projecting_ukb_ddrtree.ipynb
│   ├── distance_est.py                                 # Script to apply distance estimating algorithm required in projecting_ukb_ddrtree.ipynb
│   └── README.md                                       # This README file
```
