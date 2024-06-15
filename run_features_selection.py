import sys
from feature_selection.experiment_hub import (
    FeatureSelectionExperimentHub,
    parameter_config,
)
from data.load_data import load_data_from_csv, date_based_split

if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "jpn"
    dataset_filename = sys.argv[2] if len(sys.argv) > 2 else "jpn"
    method_name = (
        sys.argv[3]
        if len(sys.argv) > 3 and sys.argv[3] in parameter_config
        else "mrmr_filter"
    )
    nrows = int(sys.argv[4]) if len(sys.argv) > 4 else None
    random_sampling = bool(sys.argv[5]) if len(sys.argv) > 5 else False

    params = parameter_config[method_name]

    TARGET = "return_1w"
    data, features = load_data_from_csv(
        dataset_filename, nrows, None, TARGET, random_sample=random_sampling
    )
    x_train, y_train, x_validation, y_validation, x_test, y_test = date_based_split(
        df=data,
        target_column=TARGET,
        test_size=0.2,
        validation_size=0.25,
        show=False,
        day_of_week=2,
    )

    hub = FeatureSelectionExperimentHub(
        dataset_name=dataset_name,
    )

    print(f"Running {method_name} on {dataset_name} with {params}.")

    features, evaluations, runtime = hub.run_single_selection(
        method_name,
        X=x_train,
        y=y_train,
        X_val=x_validation,
        y_val=y_validation,
        parameters=params,
    )
    print(f"Final evaluations: {evaluations}")
    print(f"Total runtime: {runtime}")
