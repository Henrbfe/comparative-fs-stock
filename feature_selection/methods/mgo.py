import math
import pickle
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from numpy.random import default_rng
from feature_selection.optim_problems import SVRFeatureSelection
from feature_selection.methods.selector_interface import SelectorInterface

BASE_PATH = Path(__file__).parent.resolve()


def init_population(
    lb: float, ub: float, dim: int, pop_size: int, fobj: callable, seed: int
):
    rng = default_rng(seed)
    pop = rng.uniform(lb, ub, (pop_size, dim))
    pop_fitness = np.repeat(np.inf, pop_size)
    for i in range(pop_size):
        fitness = fobj(pop[i])
        pop_fitness[i] = fitness
    return pop, pop_fitness


def mgo(
    pop_size: int,
    max_iter: int,
    lb: float,
    ub: float,
    dim: int,
    fobj: callable,
    seed: int,
    dataset_name: str,
    early_stopping: int,
):
    """Perform the Mountain Gazelle Optimizer algorithm"""

    # Init random population
    pop, pop_fitness = init_population(lb, ub, dim, pop_size, fobj, seed)

    idx_sorted = np.argsort(pop_fitness)
    pop = pop[idx_sorted, :]
    pop_fitness = pop_fitness[idx_sorted]
    best_sol_history = []
    best_i = 0
    global_best_sol = pop[0]
    global_best_fit = np.inf
    for iteration in range(max_iter):
        print(f"Running iteration {iteration}")
        for i in range(pop_size):
            best_sol = global_best_sol  # pop[0]
            best_fitness = global_best_fit  # pop_fitness[0]

            a = -(1 + iteration / max_iter)
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand(dim)
            r4 = np.random.rand(dim)
            r6 = np.random.rand()
            ri1 = np.random.randint(1, 3)
            ri2 = np.random.randint(1, 3)
            ri3 = np.random.randint(1, 3)
            ri4 = np.random.randint(1, 3)
            ri5 = np.random.randint(1, 3)
            ri6 = np.random.randint(1, 3)
            n1 = np.random.normal(dim)
            n2 = np.random.normal(dim)
            n3 = np.random.normal(dim)
            n4 = np.random.normal(dim)

            coef = np.zeros((4, dim))
            coef[0, :] = a + 1 + r3
            coef[1, :] = a * n2
            coef[2, :] = r4
            coef[3, :] = n3 * n4**2 * np.cos((r4 * 2) * n3)

            rand_idx = np.random.randint(pop_size, size=int(np.ceil(pop_size / 3)))
            rand_sol = pop[np.random.randint(np.ceil(pop_size / 3), pop_size)]

            bh = (
                rand_sol * r1 + pop[rand_idx].mean(axis=0) * r2
            )  # Source has floor and ceil functions on r-numbers, not sure why they are there?
            f = n1 * math.exp(2 - 2 * (iteration / max_iter))
            d = (pop[i] + np.abs(best_sol)) * (2 * r6 - 1)

            tsm = (
                best_sol
                - np.abs((ri1 * bh - ri2 * pop[i]) * f)
                * coef[np.random.randint(1, 4), :]
            )
            mh = (
                bh
                + coef[np.random.randint(1, 4), :]
                + (ri3 * best_sol - ri4 * rand_sol) * coef[np.random.randint(1, 4), :]
            )
            bmh = (
                pop[i]
                - d
                + (ri5 * best_sol - ri6 * bh) * coef[np.random.randint(1, 4), :]
            )
            msf = (ub - lb) * np.random.rand(dim) + lb

            # Make sure solutions are within boundaries
            tsm[tsm > ub] = ub
            tsm[tsm < lb] = lb
            mh[mh > ub] = ub
            mh[mh < lb] = lb
            bmh[bmh > ub] = ub
            bmh[bmh < lb] = lb

            # Add new solutions to the population/herd
            new_sols = np.array([tsm, bmh, mh, msf])
            new_fitness = np.array([fobj(tsm), fobj(bmh), fobj(mh), fobj(msf)])
            pop = np.append(pop, new_sols, axis=0)
            pop_fitness = np.append(pop_fitness, new_fitness)

            # See if one of the new solution is the global best
            potential_best = new_sols + [best_sol]
            best_sol = potential_best[np.argmin(new_fitness + [best_fitness])]

        idx_sorted = np.argsort(pop_fitness)[:pop_size]
        pop = pop[idx_sorted, :]
        pop_fitness = pop_fitness[idx_sorted]
        best_sol_history.append(pop_fitness[0])
        if pop_fitness[0] < global_best_fit:
            global_best_fit = pop_fitness[0]
            global_best_sol = pop[0]
            best_i = iteration
        elif iteration > best_i + early_stopping:
            if iteration < 30:
                print("Restart population")
                pop, pop_fitness = init_population(
                    lb, ub, dim, pop_size, fobj, seed=None
                )
                best_i = max(iteration - int(early_stopping / 2), 1)
            else:
                break

    print(f"Best solution history: {best_sol_history}")
    with open(
        f"{BASE_PATH}/../saves/mgo_best_sol_history_{dataset_name}_{max_iter}.pkl", "wb"
    ) as file:
        pickle.dump(best_sol_history, file)
    plt.plot(best_sol_history)
    plt.savefig(
        f"{BASE_PATH}/../plots/mgo_best_sol_history_{dataset_name}_{max_iter}.png"
    )

    return global_best_sol, global_best_fit


class MGOSelector(SelectorInterface):

    def __init__(self, dataset_name="nasnor", estimator=SVRFeatureSelection) -> None:
        super().__init__(estimator)
        self.dataset_name = dataset_name
        self.population_size = 30
        self.alpha = 0.99
        self.algo_lb = 0.0
        self.algo_ub = 1.0
        self.seed = None
        self.max_iters = 100
        self.early_stopping = 50

    def __call__(
        self, X: DataFrame, y: Series, X_val: DataFrame, y_val: Series, parameters: dict
    ) -> list[str]:
        start_time = time.time()
        population_size = (
            parameters["population_size"]
            if "population_size" in parameters
            else self.population_size
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
        alpha = parameters["alpha"] if "alpha" in parameters else self.alpha
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
        dim = X_numpy.shape[1]
        features, fitness = mgo(
            population_size,
            max_iters,
            self.algo_lb,
            self.algo_ub,
            dim,
            model.evaluate,
            seed,
            self.dataset_name,
            early_stopping,
        )
        features = features > ((self.algo_ub - self.algo_lb) / 2)
        self.time = time.time() - start_time
        features = (
            features
            if isinstance(X, np.ndarray)
            else X.drop(columns=["gvkey", "datadate"]).columns[features].values
        )
        return features
