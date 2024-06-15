import numpy as np
import matplotlib.pyplot as plt
from feature_selection.feature_sets import categorize_features, FEATURE_SETS


def plot_features_category_counts_update(dataset, subsets):
    category_counts = {}
    colors = [(213/256, 223/256, 124/256),
            (157/256, 183/256, 225/256),
            (244/256, 172/256, 103/256),
            (200/256, 113/256, 172/256)]
    for subset in subsets:
        category_counts[subset] = {}
        print(f"-----{subset}-----")
        tech, fund, macro, desc = categorize_features(FEATURE_SETS[dataset][subset], dataset)
        category_counts[subset]["Technical"] = len(tech)
        category_counts[subset]["Fundamental"] = len(fund)
        category_counts[subset]["Macro"] = len(macro)
        category_counts[subset]["Descriptive"] = len(desc)
    # Extracting groups and categories
    groups = list(category_counts.keys())
    categories = list(category_counts[groups[0]].keys())

    # Extracting values
    values = {category: [category_counts[group][category] for group in groups] for category in categories}

    # Number of groups and categories
    n_groups = len(groups)
    n_categories = len(categories)

    # Creating bar positions
    bar_width = 0.2
    index = np.arange(n_groups)

    # Plotting
    fig, ax = plt.subplots()

    for i, category in enumerate(categories):
        bar = ax.bar(index + i * bar_width, values[category], bar_width, label=category, color=colors[i])
        for rect in bar:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

    # Labeling
    # ax.set_xlabel('Groups')
    group_names = {
        "all": "All",
        "mrmr_filter": "mRMR",
        "var_filter": "Var",
        "forward": "SFS",
        "backward": "SBE",
        "pso": "PSO",
        "ga": "GA",
        "mgo": "MGO"
    }
    dataset_names = {
        "jpn": "Japan",
        "usa": "USA",
        "nasnor": "Nasnor"
    }

    ax.set_ylabel('Number of features')
    ax.set_title(f'Feature categories for feature subset on {dataset_names[dataset]}')
    ax.set_xticks(index + bar_width * (n_categories / 2 - 0.5))
    ax.set_xticklabels([group_names[group] for group in groups], rotation=45)
    ax.legend()

    # Displaying the plot
    plt.show()
