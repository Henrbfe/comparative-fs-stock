"""
Feature selection methods using binary optmization techniques.
"""

from pathlib import Path
import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization, GeneticAlgorithm
from niapy.algorithms.basic.ga import (
    tournament_selection,
    uniform_crossover,
    uniform_mutation,
)
from feature_selection.methods.selector_interface import SelectorInterface
from feature_selection.optim_problems import SVRFeatureSelection

BASE_PATH = Path(__file__).parent.resolve()


def run_with_early_stopping(algorithm, task, early_stopping):
    algorithm.callbacks.before_run()
    pop, fpop, params = algorithm.init_population(task)
    xb, fxb = algorithm.get_best(pop, fpop)
    i = 0
    best_i = 0
    current_best = np.inf
    while not task.stopping_condition():  # for each itr
        algorithm.callbacks.before_iteration(pop, fpop, xb, fxb, **params)
        pop, fpop, xb, fxb, params = algorithm.run_iteration(
            task, pop, fpop, xb, fxb, **params
        )
        algorithm.callbacks.after_iteration(pop, fpop, xb, fxb, **params)
        task.next_iter()
        algorithm.callbacks.after_run()
        if fxb < current_best:
            best_i = i
            current_best = fxb
        if i > best_i + early_stopping:
            break
        i += 1
    return xb, fxb


class ParticleSwarm(SelectorInterface):
    def __init__(
        self, dataset_name="nasnor", estimator=SVRFeatureSelection, population_size=20
    ) -> None:
        """Initialises Particle Swarm

        Args:
            model: Class with _evaluate function. Inherits from niapy.problems.Problem
            population_size: Number of
        """
        super().__init__(estimator)
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.inertia = 1.0
        self.min_velocity = -np.inf
        self.max_velocity = np.inf
        self.alpha = 0.99
        self.seed = None
        self.max_iters = 100
        self.early_stopping = 50

    def save_and_plot_convergence_data(self, x, fitness, n_iters):
        print(f"Best solution history: {fitness}")
        plt.plot(x, fitness)
        plt.savefig(
            f"{BASE_PATH}/../plots/pso_convergence_{self.dataset_name}_{n_iters}.png"
        )
        with open(
            f"{BASE_PATH}/../saves/pso_best_sol_history_{self.dataset_name}_{n_iters}.pkl",
            "wb",
        ) as file:
            pickle.dump(fitness, file)

    def __call__(self, X, y, X_val, y_val, parameters):
        """Runs particle swarm algorithm

        Args:
            X: Training set X
            y : Training set y
            max_iters: Maximum number of iterations. Defaults to 100.

        Returns:
            best_features [numpy.ndarray]: best features,
            best_fitness [numpy.float64]: best fitness
        """
        start_time = time.time()
        population_size = (
            parameters["population_size"]
            if "population_size" in parameters
            else self.population_size
        )
        alpha = parameters["alpha"] if "alpha" in parameters else self.alpha
        inertia = parameters["w"] if "w" in parameters else self.inertia
        min_velocity = (
            parameters["min_v"] if "min_v" in parameters else self.min_velocity
        )
        max_velocity = (
            parameters["max_v"] if "max_v" in parameters else self.max_velocity
        )
        seed = parameters["seed"] if "seed" in parameters else self.seed
        max_iters = (
            parameters["max_iters"] if "max_iters" in parameters else self.max_iters
        )
        early_stopping = (
            parameters["early_stopping"]
            if "early_stopping" in parameters
            else self.early_stopping
        )
        print(f"Running {max_iters} iterations")

        algorithm = ParticleSwarmOptimization(
            population_size=population_size,
            w=inertia,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            seed=seed,
        )
        X_numpy = (
            X
            if isinstance(X, np.ndarray)
            else X.drop(columns=["gvkey", "datadate"]).to_numpy()
        )
        y_numpy = y if isinstance(y, np.ndarray) else y.to_numpy()
        X_val_numpy = (
            X_val
            if isinstance(X_val, np.ndarray)
            else X_val.drop(columns=["gvkey", "datadate"]).to_numpy()
        )
        y_val_numpy = y_val if isinstance(y_val, np.ndarray) else y_val.to_numpy()
        info_df = X_val[["gvkey", "datadate"]]
        model = self.estimator(
            X_numpy,
            y_numpy,
            X_val_numpy,
            y_val_numpy,
            info_df,
            alpha,
            self.dataset_name,
        )
        task = Task(model, max_iters=max_iters)
        # features, fitness = algorithm.run(task)
        features, fitness = run_with_early_stopping(
            algorithm=algorithm, task=task, early_stopping=early_stopping
        )
        x, fitness_history = task.convergence_data()
        self.save_and_plot_convergence_data(x, fitness_history, max_iters)
        features = features > 0.5
        self.time = time.time() - start_time
        features = (
            features
            if isinstance(X, np.ndarray)
            else X.drop(columns=["gvkey", "datadate"]).columns[features].values
        )
        return features


class GeneticAlgorithmSelector(SelectorInterface):

    def __init__(self, dataset_name="nasnor", estimator=SVRFeatureSelection) -> None:
        super().__init__(estimator)
        self.dataset_name = dataset_name
        self.population_size = 20
        self.tournament_size = 10
        self.mutation_rate = 0.25
        self.crossover_rate = 0.25
        self.selection = tournament_selection
        self.crossover = uniform_crossover
        self.mutation = uniform_mutation
        self.alpha = 0.99
        self.seed = None
        self.max_iters = 500
        self.early_stopping = 50

    def save_and_plot_convergence_data(self, x, fitness, n_iters):
        print(f"Best solution history: {fitness}")
        plt.plot(x, fitness)
        plt.savefig(
            f"{BASE_PATH}/../plots/ga_convergence_{self.dataset_name}_{n_iters}.png"
        )
        with open(
            f"{BASE_PATH}/../saves/ga_best_sol_history_{self.dataset_name}_{n_iters}.pkl",
            "wb",
        ) as file:
            pickle.dump(fitness, file)

    def __call__(self, X, y, X_val, y_val, parameters):
        start_time = time.time()
        population_size = (
            parameters["population_size"]
            if "population_size" in parameters
            else self.population_size
        )
        tournament_size = (
            parameters["tournament_size"]
            if "tournament_size" in parameters
            else self.tournament_size
        )
        mutation_rate = (
            parameters["mutation_rate"]
            if "mutation_rate" in parameters
            else self.mutation_rate
        )
        crossover_rate = (
            parameters["crossover_rate"]
            if "crossover_rate" in parameters
            else self.crossover_rate
        )
        selection = (
            parameters["selection"] if "selection" in parameters else self.selection
        )
        crossover = (
            parameters["crossover"] if "crossover" in parameters else self.crossover
        )
        mutation = parameters["mutation"] if "mutation" in parameters else self.mutation
        alpha = parameters["alpha"] if "alpha" in parameters else self.alpha
        seed = parameters["seed"] if "seed" in parameters else self.seed
        max_iters = (
            parameters["max_iters"] if "max_iters" in parameters else self.max_iters
        )
        early_stopping = (
            parameters["early_stopping"]
            if "early_stopping" in parameters
            else self.early_stopping
        )
        print(f"Running {max_iters} iterations")

        algorithm = GeneticAlgorithm(
            population_size=population_size,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            seed=seed,
        )
        X_numpy = (
            X
            if isinstance(X, np.ndarray)
            else X.drop(columns=["gvkey", "datadate"]).to_numpy()
        )
        y_numpy = y if isinstance(y, np.ndarray) else y.to_numpy()
        X_val_numpy = (
            X_val
            if isinstance(X_val, np.ndarray)
            else X_val.drop(columns=["gvkey", "datadate"]).to_numpy()
        )
        y_val_numpy = y_val if isinstance(y_val, np.ndarray) else y_val.to_numpy()
        info_df = X_val[["gvkey", "datadate"]]
        model = self.estimator(
            X_numpy,
            y_numpy,
            X_val_numpy,
            y_val_numpy,
            info_df,
            alpha,
            self.dataset_name,
        )
        task = Task(model, max_iters=max_iters)
        # features, fitness = algorithm.run(task)
        features, fitness = run_with_early_stopping(
            algorithm=algorithm, task=task, early_stopping=early_stopping
        )
        x, fitness_history = task.convergence_data()
        self.save_and_plot_convergence_data(x, fitness_history, max_iters)
        features = features > 0.5
        self.time = time.time() - start_time
        features = (
            features
            if isinstance(X, np.ndarray)
            else X.drop(columns=["gvkey", "datadate"]).columns[features].values
        )
        return features
