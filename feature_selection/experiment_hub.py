"""
Hub for running feature selection experiments.
"""

from typing import Callable
import pandas as pd
import pickle
from pathlib import Path
from seaborn import heatmap
from niapy.algorithms.basic.ga import (
    tournament_selection,
    uniform_crossover,
    uniform_mutation,
)
from feature_selection.methods.filter import (
    VarianceFilterSelector,
    MRMRFilterSelector,
)
from feature_selection.methods.mgo import MGOSelector
from feature_selection.methods.optimizers import (
    ParticleSwarm,
    GeneticAlgorithmSelector,
)
from feature_selection.methods.sequential import BackwardSelector, ForwardSelector
from feature_selection.optim_problems import SVRFeatureSelection
from feature_selection.methods.selector_interface import SelectorInterface


BASE_PATH = Path(__file__).parent.resolve()

parameter_config = {
    "var_filter": {"threshold": 0.35},
    "mrmr_filter": {"n_features": 50, "rel_method": "f", "red_method": "c"},
    "forward": {"cv": 2, "n_features_to_select": 30, "tol": 1e-8},
    "backward": {"cv": 2, "n_features_to_select": 30, "tol": 1e-8},
    "pso": {
        "population_size": 20,
        "c1": 3.0,  # Cognitive component
        "c2": 2.0,  # Social component
        "w": 0.7,  # Inertial weight
        "min_v": -3.0,  # Minimal velocity
        "max_v": 3.0,  # Maximal velocity
        "seed": 123,
        "alpha": 0.5,
        "max_iters": 500,
    },
    "ga": {
        "population_size": 20,
        "tournament_size": 5,
        "mutation_rate": 0.35,
        "crossover_rate": 0.25,
        "selection": tournament_selection,
        "crossover": uniform_crossover,
        "mutation": uniform_mutation,  # Look in niapy.algorithms.basic.ga for more available selection/crossover/mutation
        "seed": 123,
        "alpha": 0.5,
        "max_iters": 500,
    },
    "mgo": {
        "population_size": 20,
        "seed": 123,
        "alpha": 0.5,
        "max_iters": 500,
    },
}

TRAIN_TEST_SPLIT = 4 / 5


class FeatureSelectionExperimentHub:
    """
    Class used to run feature selection experiments.
    """

    def __init__(
        self,
        parameters=parameter_config,
        estimator_name="svr",
        fmodel=SVRFeatureSelection,
        dataset_name="nasnor",
    ):
        self.methods: dict[str, SelectorInterface] = {
            "var_filter": VarianceFilterSelector(),
            "mrmr_filter": MRMRFilterSelector(),
            "forward": ForwardSelector(
                dataset_name=dataset_name,
                estimator_name=estimator_name,
            ),
            "backward": BackwardSelector(
                dataset_name=dataset_name,
                estimator_name=estimator_name,
            ),
            "pso": ParticleSwarm(dataset_name=dataset_name, estimator=fmodel),
            "ga": GeneticAlgorithmSelector(dataset_name=dataset_name, estimator=fmodel),
            "mgo": MGOSelector(dataset_name=dataset_name, estimator=fmodel),
        }
        self.parameters = parameters

    def run_single_selection(
        self,
        method_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        parameters: dict,
    ):
        """
        Run a single feature selection with the given method.
        Configuration other than default may be provided as keyword arguments.

        returns:
            Selected features, final evaluation, and runtime.
        """

        if method_name not in self.methods:
            return ValueError(f"Method with name {method_name} does not exist.")
        method = self.methods[method_name]
        features = method(X, y, X_val, y_val, parameters)
        print(f"---------- Features found for {method_name} ----------")
        print(features)
        evaluations = []
        runtime = method.get_runtime()
        return features, evaluations, runtime

    def heatmap_results(self, results, savefile: str):
        eval_data = {}
        runtime_data = {}
        for dataset in results:
            eval_data[dataset] = {}
            runtime_data[dataset] = {}
            for method in results[dataset]:
                eval_data[dataset][method] = results[dataset][method]["evaluations"][0]
                runtime_data[dataset][method] = results[dataset][method]["runtime"]
        eval_plot = heatmap(pd.DataFrame(eval_data))
        figure = eval_plot.get_figure()
        figure.savefig(f"{BASE_PATH}/plots/{savefile}_eval_heatmap.png")
        runtime_plot = heatmap(pd.DataFrame(runtime_data))
        figure = runtime_plot.get_figure()
        figure.savefig(f"{BASE_PATH}/plots/{savefile}_runtime_heatmap.png")

    def run_all_selections(
        self,
        data: pd.DataFrame,
        name: str,
        features: list[str],
        target: str,
        save_file=None,
    ):
        """
        Run all available feature selection methods with the run_single_selection method.

        returns:
            A dictionary with method names as keys and selected features, final evaluation,
            and runtime as values.
        """

        results = {}
        split_index = int(len(data) * TRAIN_TEST_SPLIT)
        training_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        X = training_data[features]
        y = training_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        print(f"Running on {name} data")
        method_runs = {}
        for method in self.methods:
            print(f"Beginning {method}")
            features, evaluations, runtime = self.run_single_selection(
                method, X, y, X_test, y_test, self.parameters[method]
            )
            method_runs[method] = {
                "selection": features,
                "evaluations": evaluations,
                "runtime": runtime,
            }
            print(f"Finished {method}")
        results[name] = method_runs
        if save_file:
            ds_filepath = f"{BASE_PATH}/saves/{save_file}_{name}.pkl"
            with open(ds_filepath, "wb") as file:
                pickle.dump(results, file)

        self.heatmap_results(results, save_file)

        save_file = save_file if save_file else f"all_selections_{name}_{len(data)}"
        filepath = f"{BASE_PATH}/saves/{save_file}.pkl"
        with open(filepath, "wb") as file:
            pickle.dump(results, file)
            print(f"Results saved to {filepath}")
        return results
