# Pipeline for creating stock datasets, applying feature selection, and simulate stock portfolios using ML models.
This project is made for the Master thesis at NTNU:
_Comparative analysis of feature selection in stock price prediction_
In case of issues, please submit to the public Github. Note that the APIs of WRDS and IMF appear to be unstable at times, notify maintainers if there are issues with dataset creation.

## Structure
- Data: create datasets for a given country
- Feature selection: implement and run feature selection methods on datasets created in Data
- Model: define and train ML models
- Evaluation: run simulations and calculate evaluation metrics for ML models

- results: folder containing result files from running simulations

## Scripts
All scripts should be run from root.

- run_create_dataset.py

    ```python run_create_dataset.py FRA 2020-01-01 2024-01-01```

- run_feature_selection.py

    ```python run_feature_selection.py fra datasets/FRA_2022-01-01_2024-01-01.csv mrmr_filter```

- run_hyperparameter_tuning.py

    ```python run_hyperparameter_tuning.py svr_nystroem fra datasets/FRA_2022-01-01_2024-01-01.csv mrmr_filter```

- run_simulation_suite.py

    ```python run_simulation_suite.py fra datasets/FRA_2022-01-01_2024-01-01.csv mrmr_filter svr```

## Workflow

- Create datasets with the ```run_create_dataset.py``` script.
- Run feature selections with the ```run_feature_selection.py``` script.
- Add hyperparams in ```model/hyperparameters.py``` for svr used for evaluation in feature selection. These parameters can be obtained by running hyperparameter tuning using the all features set.
- Add feature sets to the ```feature_selection/feature_sets.py``` file.
- Run hyperparameter tuning (if desired) with the ```run_hyperparameter_tuning.py``` script.
- Add hyperparameters for the dataset in ```model/hyperparameters.py```.
- Run simulation suite to evaluate a given feature set with the ````run_simulation_suite.py``` script. The results are stored in the ```results/``` folder.
