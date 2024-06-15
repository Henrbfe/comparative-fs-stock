import statistics
import seaborn as sn
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew, kurtosis, shapiro
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def get_max_from_array(array, k=1):
    """get max values and indices from array.
    Args:
        array: array to pick from
        k: number of entries. Defaults to 1.
    Returns:
        values, indices
    """
    array = array.reshape(-1)
    sorted_indices = np.argsort(array, axis=-1)
    topk_indices = np.take_along_axis(sorted_indices, np.arange(-k, 0), axis=-1)
    topk_values = np.take_along_axis(array, topk_indices, axis=-1)
    return topk_values, topk_indices


def get_min_from_array(array, k=1):
    """get min values and indices from array.
    Args:
        array: array to pick from
        k: number of entries. Defaults to 1.
    Returns:
        values, indices
    """
    array = array.reshape(-1)
    sorted_indices = np.argsort(array, axis=-1)
    topk_indices = np.take_along_axis(sorted_indices, np.arange(k), axis=-1)
    topk_values = np.take_along_axis(array, topk_indices, axis=-1)
    return topk_values, topk_indices


def get_long_short_from_array(pred_array, value_array, k):
    """gets mean return for a long-short portfolio.

    Args:
        pred_array: prediction array
        value_array: value array
        k: number of shorts and longs

    Returns:
        average return
    """
    top_values, top_indices = get_max_from_array(pred_array, k)
    bottom_values, bottom_indices = get_min_from_array(pred_array, k)
    return np.mean(value_array.reshape(-1)[top_indices]) - np.mean(
        value_array.reshape(-1)[bottom_indices]
    )


def get_long_from_array(pred_array, value_array, k):
    """gets mean return for a long position.
    Args:
        pred_array: prediction array
        value_array: value array
        k: number of shorts and longs
    Returns:
        average return
    """
    top_values, top_indices = get_max_from_array(pred_array, k)
    return np.mean(value_array.reshape(-1)[top_indices])


def get_short_from_array(pred_array, value_array, k):
    """gets mean return for a short position. Ideally negative
    Args:
        pred_array: prediction array
        value_array: value array
        k: number of shorts and longs
    Returns:
        average return
    """
    bottom_values, bottom_indices = get_min_from_array(pred_array, k)
    return np.mean(value_array.reshape(-1)[bottom_indices])


def get_return_quantiles(
    pred_array,
    value_array,
    number_of_quantiles,
    truncate_axis=True,
    boxplot=False,
    show=True,
    save_name=None,
):
    """Creates graph with predicted return quantiles and their actual return

    Args:
        pred_array: prediction array
        value_array: value array
        number_of_quantiles: number of buckets to split the data in.
        truncate_axis: Defaults to True.
        boxplot: boolean to add boxplot. Defaults to False.
        show: boolean to dispaly graph. Defaults to True.
        save_name: name of path to save graph in. Defaults to None.
    """
    indices = np.argsort(pred_array.reshape(-1))
    split = [
        value_array.reshape(-1)[quant]
        for quant in np.array_split(indices, number_of_quantiles)
    ]
    averages = [np.mean(arr) for arr in split]
    plt.bar(range(1, len(split) + 1), averages, color="skyblue")
    if boxplot:
        plt.boxplot(split, vert=True)
    plt.xlabel("Quantile")
    plt.ylabel("Return")
    plt.title("Return per quantile")
    if truncate_axis:
        plt.ylim(min(averages) * 0.85, max(averages) * 1.15)
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()


def get_positive_and_negative_return_matrix(
    pred_array, value_array, title, save_name=None, show=True
):
    """generates return matrix for postive and negative return

    Args:
        pred_array: prediction array
        value_array: value array
        title: name of graph
        save_name: name of path to save graph in. Defaults to None.
        show: boolean to dispaly graph. Defaults to True.
    """
    get_return_matrix(
        (value_array > 1.0).astype(int),
        (pred_array > 1.0).astype(int),
        2,
        title,
        save_name,
        show,
    )


def get_return_matrix_per_quantile(
    pred_array, value_array, number_of_quantiles, title, save_name=None, show=True
):
    """generates return matrix for set size of quantiles

    Args:
        pred_array: prediction array
        value_array: value array
        number_of_quantiles: number of buckets to split data into
        title: name of graph
        save_name: name of path to save graph in. Defaults to None.
        show: boolean to dispaly graph. Defaults to True.
    """
    y_pred = np.full(len(pred_array), -1, dtype=int)
    y_true = np.full(len(value_array), -1, dtype=int)
    indices_pred = np.array_split(
        np.argsort(pred_array.reshape(-1)), number_of_quantiles
    )
    indices_true = np.array_split(
        np.argsort(value_array.reshape(-1)), number_of_quantiles
    )
    assert len(indices_pred) == len(indices_true), "List must have the same length!"
    for index in range(number_of_quantiles):
        y_pred[indices_pred[index]] = index
        y_true[indices_true[index]] = index
    get_return_matrix(y_true, y_pred, number_of_quantiles, title, save_name, show)


def get_return_matrix(
    y_true, y_pred, number_of_quantiles, title, save_name=None, show=True
):
    """generates a return matrix based on inputs.

    Args:
     y_true: actual values
     y_pred: predicted values
     number_of_quantiles: number of buckets to split data into. Only used for naming
     title: name of graph
     save_name: name of path to save graph in. Defaults to None.
     show: boolean to dispaly graph. Defaults to True.
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[f"actual_{i}" for i in range(0, number_of_quantiles)],
        columns=[f"pred_{i}" for i in range(0, number_of_quantiles)],
    )
    sn.heatmap(
        df_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidth=0.5,
        annot_kws={"size": 12, "color": "red"},
        cbar=False,
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()


def get_pred_to_return_scatter(
    pred, actual, title, interval=[0, 2], save_name=None, show=True
):
    """
    Generates a scatter plot of the actual values against the predicted values

    Args:
        pred: predicted values
        actual: true value
        title: name of graph
        interval: data interval for allowed values. Defaults to [0, 2].
        save_name: name of path to save graph in. Defaults to None.
        show: boolean to dispaly graph. Defaults to True.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual.clip(min=interval[0], max=interval[1]),
        pred.clip(min=interval[0], max=interval[1]),
        color="lightblue",
        alpha=0.6,
        s=12,
    )
    plt.scatter(
        actual.clip(min=interval[0], max=interval[1]),
        pred.clip(min=interval[0], max=interval[1]),
        color="darkblue",
        alpha=0.6,
        s=6,
    )
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)

    # Plotting the line of best fit (optional)
    m, b = np.polyfit(actual, pred, 1)
    plt.plot(
        actual,
        m * actual + b,
        color="red",
        linestyle="dashed",
        label="current best fit",
    )
    plt.plot(actual, actual, color="black", linestyle="dashed", label="Optimal fit")
    plt.legend()

    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()


def create_histograms(
    pred_array,
    actual_array,
    title,
    bins=50,
    interval=[0.7, 1.3],
    save_name=None,
    show=True,
):
    """method for generating histograms of predicted and actual values

    Args:
        pred_array: predicted values
        actual_array: actual values
        title: title of graph
        bins: number of bins to split data into. Defaults to 50.
        interval: data interval for allowed values. Defaults to [0.7, 1.3].
        save_name: name of path to save graph in. Defaults to None.
        show: boolean to dispaly graph. Defaults to True.
    """
    pred = (
        pred_array.clip(min=interval[0], max=interval[1])
        if isinstance(pred_array, np.ndarray)
        else np.array(pred_array).clip(min=interval[0], max=interval[1])
    )
    actual = (
        actual_array.clip(min=interval[0], max=interval[1])
        if isinstance(actual_array, np.ndarray)
        else np.array(actual_array).clip(min=interval[0], max=interval[1])
    )
    plt.hist(
        pred, bins=bins, range=interval, label="Prediction", alpha=0.5, density=True
    )
    plt.hist(
        actual,
        bins=bins,
        range=interval,
        label="Actual return",
        alpha=0.5,
        density=True,
    )
    plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    sn.kdeplot(pred, bw=0.5)
    sn.kdeplot(actual, bw=0.5)
    plt.legend(labels=["Prediction", "Actual Return", "Prediction line", "Return line"])

    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()

def display_train_val_test_difference(
    pred_train_y,
    train_y,
    pred_val_y,
    val_y,
    pred_test_y,
    test_y,
    save_name=None,
    show=False,
):
    """Displays accuracy based metrics 

    Args:
        pred_train_y: predictions for train dataset
        train_y: targets for train dataset
        pred_val_y: predictions for validation dataset
        val_y: targets for validation dataset
        pred_test_y: predictions for test dataset
        test_y: targets for test dataset
        save_name: Save name for graph. Defaults to None.
        show: Boolean to display graph or not Defaults to False.

    Returns:
        _type_: _description_
    """
    (
        train_accuracy,
        (_, train_pos_rec, train_pos_prec),
        (_, train_neg_rec, train_neg_prec),
        _,
    ) = get_accuracy_based_metrics(pred_train_y, train_y, 1)
    val_accuracy, (_, val_pos_rec, val_pos_prec), (_, val_neg_rec, val_neg_prec), _ = (
        get_accuracy_based_metrics(pred_val_y, val_y, 1)
    )
    (
        test_accuracy,
        (_, test_pos_rec, test_pos_prec),
        (_, test_neg_rec, test_neg_prec),
        _,
    ) = get_accuracy_based_metrics(pred_test_y, test_y, 1)

    train_metrics = [
        train_accuracy,
        train_pos_rec,
        train_pos_prec,
        train_neg_rec,
        train_neg_prec,
    ]
    val_metrics = [val_accuracy, val_pos_rec, val_pos_prec, val_neg_rec, val_neg_prec]
    test_metrics = [
        test_accuracy,
        test_pos_rec,
        test_pos_prec,
        test_neg_rec,
        test_neg_prec,
    ]

    labels = ["Accuracy", "Pos Recall", "Pos Precision", "Neg Recall", "Neg Precision"]
    pos = list(range(len(train_metrics)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    plt.bar(pos, train_metrics, width, alpha=0.5, color="g", label="Train")
    plt.bar(
        [p + width for p in pos], val_metrics, width, alpha=0.5, color="b", label="Val"
    )
    plt.bar(
        [p + width * 2 for p in pos],
        test_metrics,
        width,
        alpha=0.5,
        color="r",
        label="Test",
    )

    ax.set_xticks([p + 1.5 * width for p in pos])
    ax.set_xticklabels(labels)

    plt.legend(["Train", "Val", "Test"], loc="upper left")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Comparison of Metrics")

    plt.grid()
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()
    return train_metrics, val_metrics, test_metrics


def plot_simulation(graphs, title, show, save_name):
    """Method for plotting simulation

    Args:
        dates: list of dates
        graphs: dictionary with (m_name,color,linestyle): (list(dates), list (data)),
        buy_sell_percentage: What percentage that is bought and sold. Used in title
        show_graph: Whether or not to show graph.
        save_name_for_graph: Name to store the graph as. False to not store the graph.
    """
    for label, values in graphs.items():
        plt.plot(
            values[0], values[1], label=label[0], color=label[1], linestyle=label[2]
        )
    plt.legend()
    plt.ylabel("Returns")
    plt.xlabel("Dates")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.clf()


def calculate_d_variance(mean, st_dev, skew, kurt, quantile):
    """method for calculating d-variance.
    Based on the following paper: https://doi.org/10.1016/j.eswa.2022.116970
    https://github.com/JDE65/D-ratio/blob/main/d_ratio.py


    Args:
        mean: distribution's mean
        st_dev: distribution's standard deviation
        skew: distribution's skewness
        kurt: distribution's kurtosis
        quantile: quantile

    Returns:
        cf_exp: The adjusted quantile value based on the original quantile
        cf_var: Cornish-Fisher variance
    """
    cf_exp = quantile + (quantile**2 - 1) * skew / 6
    +(quantile**3 - 3 * quantile) * kurt / 24
    -(2 * quantile**3 - 5 * quantile) * (skew**2) / 36
    cf_var = mean + st_dev * cf_exp
    return cf_exp, cf_var


def get_distribution_info(log_return):
    """returns distribution information on mean, st_dev, skew and kurt

    Args:
        log_return: list of returns

    Returns:
        information about distribution.
    """
    mean = np.mean(log_return)
    st_dev = np.std(log_return)
    skew = stats.skew(log_return)
    kurt = stats.kurtosis(log_return)
    return mean, st_dev, skew, kurt


def get_d_ratio(log_return_predictions, log_return_buy_and_hold, confid):  # confid = 1%
    """
    Returns D-ratio from the following paper: https://doi.org/10.1016/j.eswa.2022.116970
    Args:
        log_return_predictions: realised return
        log_return_buy_and_hold: buy and hold strategy
        confid: confidence interval

    Returns:
        d: d-ratio
        d_ret: d-return
        d_var: d-variance
    """
    quantile = norm.ppf(confid)
    bh_mean, bh_st_dev, bh_skew, bh_kurt = get_distribution_info(
        log_return_buy_and_hold
    )
    pred_mean, pred_st_dev, pred_skew, pred_kurt = get_distribution_info(
        log_return_predictions
    )
    cf_var_bh = calculate_d_variance(bh_mean, bh_st_dev, bh_skew, bh_kurt, quantile)[1]
    cf_var_pred = calculate_d_variance(
        pred_mean, pred_st_dev, pred_skew, pred_kurt, quantile
    )[1]
    avg_ret_bh = bh_mean * 252  # 252 trading days a year
    avg_ret_pred = pred_mean * 252
    d_ret = 1 + (avg_ret_pred - avg_ret_bh) / abs(avg_ret_bh)
    d_var = cf_var_pred / cf_var_bh
    d = d_ret / d_var
    return d, d_ret, d_var  # ,avg_ret_bh, avg_ret_pred


def get_accuracy_based_metrics(pred, test_y, threshold=1.0):
    """Returns accuracy based metrics

    Args:
        pred: predicted values
        test_y: actual values
        threshold: threshold for split. default 1, indicating loss in money.

    Returns:
        accuracy, (pos_f1, pos_recall, pos_precision), (neg_f1, neg_recall, neg_precision)
    """
    # Apply the threshold to classify as over or under 1
    pred_class = np.squeeze((pred > threshold).astype(int))
    test_y_class = np.squeeze((test_y > threshold).astype(int))

    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    TP = ((pred_class == 1) & (test_y_class == 1)).sum()
    FP = ((pred_class == 1) & (test_y_class == 0)).sum()
    TN = ((pred_class == 0) & (test_y_class == 0)).sum()
    FN = ((pred_class == 0) & (test_y_class == 1)).sum()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    pos_precision = TP / (TP + FP)
    neg_precision = TN / (TN + FN)
    pos_recall = TP / (TP + FN)
    neg_recall = TN / (TN + FP)
    pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    mcc = matthews_corrcoef(test_y_class, pred_class)

    return (
        accuracy,
        (pos_f1, pos_recall, pos_precision),
        (neg_f1, neg_recall, neg_precision),
        mcc,
    )


def calculate_sharpe_ratio(values):
    """calculates sharpe ratio

    Args:
        values: portfolio-value during simulation

    Returns:
        sharpe ratio
    """
    tot_return = (values[-1] - values[0]) / values[0]
    return tot_return / statistics.stdev(values)


def calculate_calmar_ratio(values):
    """calculates calmar ratio

    Args:
        values: portfolio-value during simulation

    Returns:
        calmar ratio
    """
    tot_return = (values[-1] - values[0]) / values[0]
    if tot_return == 0:
        return 0
    drawdown = min(values) / tot_return
    return tot_return / drawdown


def get_drawdowns(values):
    """Get all drawdown metrics

    Args:
        values: portfolio-value during simulation

    Returns:
        returns  longest_drawdown_start, longest_drawdown_end,
        largest drawdown, index of largest drawdown, index before largest drawdown,
        index after largest drawdown
    """
    drawdowns = calculate_drawdowns(values)
    longest_drawdown_start, longest_drawdown_end = get_longest_recover_time(drawdowns)
    largest, index, before, after = get_largest_drawdown(drawdowns)
    return longest_drawdown_start, longest_drawdown_end, largest, index, before, after


def get_longest_recover_time(drawdowns):
    """gets longest drawdown recovery time, i.e. widest vally

    Args:
        drawdowns: list of drawdowns

    Returns:
        longest_drawdown_start, longest_drawdown_end
    """
    longest_drawdown_start = 0
    longest_drawdown_end = 0
    temp_start = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] == 0 or i == len(drawdowns):
            if i - temp_start > longest_drawdown_end - longest_drawdown_start:
                longest_drawdown_start = temp_start
                longest_drawdown_end = i
            temp_start = i
    return longest_drawdown_start, longest_drawdown_end


def get_largest_drawdown(drawdowns):
    """get maximum drawdown. i.e. deepest vally

    Args:
        drawdowns: list of drawdowns

    Returns:
        largest drawdown, index of largest drawdown,
        index before largest drawdown, index after largest drawdown
    """
    largest = max(drawdowns)
    index = drawdowns.index(largest)
    before, after = 0, len(drawdowns)
    for i in range(index, -1, -1):
        if drawdowns[i] == 0:
            before = i
            break
    for i in range(index, len(drawdowns)):
        if drawdowns[i] == 0:
            after = i
            break
    return largest, index, before, after


def calculate_drawdowns(values):
    """Calculates drawdowns from portfolio values

    Args:
        values: portfolio-value during simulation

    Returns:
        list of drawdowns
    """
    peak, trough = values[0], values[0]
    drawdowns = []
    for value in values:
        if value > peak:
            peak = value
            drawdowns.append(0)
        else:
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
            if value < trough:
                trough = value
    return drawdowns


import pandas as pd
import matplotlib.pyplot as plt

INDIVIDUAL_SETS = {
    "usa": {
        "SVR": {
            "x": "SBE",
            "y": "PSO",
            "z": "SFS"
        },
        "XGB": {
            "x": "SBE",
            "y": "MGO",
            "z": "SFS"
        }
    },
    "jpn": {
        "SVR": {
            "x": "PSO",
            "y": "SFS",
            "z": "GA"
        },
        "XGB": {
            "x": "PSO",
            "y": "SBE",
            "z": "mRMR"
        }
    },
    "nasnor": {
        "SVR": {
            "x": "MGO",
            "y": "mRMR",
            "z": "SFS"
        },
        "XGB": {
            "x": "MGO",
            "y": "mRMR",
            "z": "PSO"
        }
    }
}

def create_sub_plots_from_return_df(path, model_name, dataset_name):
    """create plot showing different strategies

    Args:
        path: path of result csv
        model_name: model name
        dataset_name: dataset name
    """
    df = pd.read_csv(path)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle("Comparison of Model Returns for Different Strategies", fontsize=36)

    titles = ["5%", "10%", "20%", "50%"]
    strategies = ["long_short5", "long_short10", "long_short20", "long_short50"]

    lines, labels = [], []

    color_dict = {
        "mrmr_filter": (255 / 256, 0 / 256, 0 / 256),  # red
        "var_filter": (0 / 256, 255 / 256, 0 / 256),  # green
        "forward": (0 / 256, 0 / 256, 255 / 256),  # blue
        "backward": (0 / 256, 255 / 256, 255 / 256),  # cyan
        "ga": (125 / 256, 125 / 256, 90 / 256),  # magenta
        "pso": (180 / 256, 180 / 256, 0 / 256),  # yellow
        "mgo": (128 / 256, 0 / 256, 128 / 256),  # purple
        "all": (0 / 256, 0 / 256, 0 / 256),  # black
        "union_x_z": (61 / 256, 54 / 256, 40 / 256),  # brown
        "union_y_z": (255 / 256, 192 / 256, 203 / 256),  # pink
        "union_x_y_z": (0 / 256, 255 / 256, 127 / 256),  # lime
        "union_x_y": (0 / 256, 128 / 256, 128 / 256),  # teal
        "inter_x_z": (61 / 256, 54 / 256, 40 / 256),  # brown
        "inter_y_z": (255 / 256, 192 / 256, 203 / 256),  # pink
        "inter_x_y_z": (0 / 256, 255 / 256, 127 / 256),  # lime
        "inter_x_y": (0 / 256, 128 / 256, 128 / 256),  # teal
    }

    for ax, strategy, title in zip(axes.flatten(), strategies, titles):
        for _, row in df.iterrows():
            counter=0 #to avoid same plots being exactly on top of each other
            if dataset_name in row["path"] and model_name in row["type"]:
                dates = row["dates"][1:-1].split(", ")
                try: 
                    dates = pd.to_datetime(dates)
                except:
                    dates = pd.to_datetime(dates, format="Timestamp('%Y-%m-%d 00:00:00')")
                values = pd.to_numeric(row[strategy][1:-1].split(", "))
                index = pd.to_numeric(row["index"][1:-1].split(", "))
                color = (0 / 256, 0 / 256, 0 / 256)
                for c_name in color_dict.keys():
                    if c_name in row["custom_name"]:
                        color = color_dict[c_name]
                        break
                (line,) = ax.plot(dates, values+counter/1000, label="_".join(row["custom_name"].split("_")[3:]), color=color, lw=2.0)
                if (
                    line.get_label() not in labels
                ):  # Check to avoid duplicates in the legend
                    lines.append(line)
                    labels.append(line.get_label())
        (index_line,) = ax.plot(dates, index, label="index", color="darkorange", lw=2.0)
        if "Market Index" not in labels:
            lines.append(index_line)
            labels.append("Market Index")
        ax.set_title(title, fontsize=28)
        #ax.set_xlabel("Date", fontsize=20)
        ax.set_ylabel("Returns", fontsize=20)
        ax.tick_params(axis='x', rotation=45)  # Tilt x-axis labels
    label_names = {
        "all": "All",
        "var_filter": "Var",
        "mrmr_filter": "mRMR",
        "forward": "SFS",
        "backward": "SBE",
        "pso": "PSO",
        "ga": "GA",
        "mgo": "MGO",
    }
    labels_display = []
    for label in labels:
        if label in label_names:
            labels_display.append(label_names[label])
        elif "union" in label or "inter" in label:
            split_element = " U " if "union" in label else " ∩ "
            components = label.split("_")[1:-1]
            labels_display.append(split_element.join([INDIVIDUAL_SETS[dataset_name][model_name][component] for component in components]))
        else:
            labels_display.append(label)
    leg = fig.legend(lines, labels_display, loc="lower center", ncol=5, fontsize=26, columnspacing=4.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(15.0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.3, hspace=0.47)
    plt.show()


def plot_metrics(filename: str, dataset: str, model: str):
    """Plots the different metrics for a model and dataset combination

    Args:
        path: path of result csv
        model_name: model name
        dataset_name: dataset name
    """
    df = pd.read_csv(filename)
    df["dataset"] = df["path"].str.split("/").str.get(3).str.split("_").str.get(1)
    df["fs"] = df["path"].str.split("/").str.get(3).str.split("_").str.get(2)
    df = df[
        ["path", "type", "dataset", "fs"]
        + df.drop(columns=["path", "type", "dataset", "fs"]).columns.values.tolist()
    ]
    df = df[(df["type"] == model) & (df["dataset"] == dataset)]
    metrics = {
        "classif": [
            "accuracy",
            "pos_f1",
            "pos_recall",
            "pos_precision",
            "neg_f1",
            "neg_recall",
            "neg_precision",
        ],
        "yields": ["long_short5", "long_short10", "long_short20", "long_short50"],
        "calmar": ["cal5", "cal10", "cal20", "cal50"],
        "sharpe": ["sharpe5", "sharpe10", "sharpe20", "sharpe50"],
        "d_ratio": ["d-ratio5", "d-ratio10", "d-ratio20", "d-ratio50"],
        "d_return": ["d-return5", "d-return10", "d-return20", "d-return50"],
        "d_var": ["d-variance5", "d-variance10", "d-variance20", "d-variance50"],
        "profit_trade": [
            "profit_per_trade5",
            "profit_per_trade10",
            "profit_per_trade20",
            "profit_per_trade50",
        ],
        "max_dd": [
            "max_drawdown5",
            "max_drawdown10",
            "max_drawdown20",
            "max_drawdown50",
        ],
        "longest_dd": [
            "longest_drawdown_period5",
            "longest_drawdown_period10",
            "longest_drawdown_period20",
            "longest_drawdown_period50",
        ],
        "largest_dd": [
            "largest_drawdown_ratio5",
            "largest_drawdown_ratio10",
            "largest_drawdown_ratio20",
            "largest_drawdown_ratio50",
        ],
        "error": ["mse", "mae", "mape", "r2_score", "MCC"],
    }

    # fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 18))
    # fig.suptitle(f"Metric values for {model} on {dataset}")

    # for i, metric_type in enumerate(metrics):
    #     ax = axes.flatten()[i]
    #     x = np.arange(len(df["fs"]))
    #     width = 0.15
    #     multiplier = 0
    #     for metric in metrics[metric_type]:
    #         offset = width * multiplier
    #         vals = df[metric].values.flatten()
    #         print(f"{metric_type}: {metric}: {vals}")
    #         rects = ax.bar(x + offset, vals, width, label=df["fs"].values)
    #         ax.bar_label(rects, padding=3)
    #     ax.set_ylabel('Value')
    #     ax.set_title(f'{metric_type}')
    #     ax.set_xticks(x + width, df["fs"].values.flatten())
    #     ax.legend(loc='upper left', ncols=len(df["fs"].values.flatten()))
    #     # ax.set_ylim(0, 250)

    # plt.show()
    fig, axes = plt.subplots(3, 4, figsize=(40, 15))
    axes = axes.flatten()
    n_fs = df['fs'].nunique()
    bar_width = 0.8 / n_fs

    # Iterate over each group and plot
    for i, group in enumerate(metrics):
        ax = axes[i]
        group_cols = metrics[group]
        x = np.arange(len(group_cols))
        for j, fs_id in enumerate(df['fs']):
            subset = df[df['fs'] == fs_id]
            bar_positions = x + j * bar_width
            ax.bar(bar_positions, subset[group_cols].values.flatten(), bar_width, label=fs_id)
        ax.set_title(f'{group}')
        ax.set_xticks(x + bar_width * (n_fs / 2 - 0.5))  # Center the xticks
        ax.set_xticklabels(group_cols)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.legend(title='fs')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(filename: str, dataset=None, model=None):
    """Plot correlatin heatmap between different output metrics

    Args:
        filename: path of result csv
        dataset: dataset name
        model: model name
    """
    df = pd.read_csv(filename)
    plt.rcParams.update({'font.size': 20})
    if dataset:
        df["fs"] = df["path"].str.split("/").str.get(3).str.split("_").str.get(1)
        df = df[df["fs"] == dataset]
    if model:
        df = df[df["type"] == model]
    df = df[df["type"].isin(["SVR", "XGB"])]
    cols = df.drop(columns=["type", "path"]).columns.values.tolist()
    blue_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", "#00509e"])
    strategies = [["5", "10"], ["20", "50"]]
    corr_mats = []
    for i, pair in enumerate(strategies):
        for j, strat in enumerate(pair):
            cols = [f"long_short{strat}", f"sharpe{strat}", "accuracy", "mae", "mse", "mape", f"d-ratio{strat}"]
            corr_mats.append(df[cols].corr().iloc[2:, :2].to_numpy())

    cols = ["Return", "Sharpe"]
    indices = ["Accuracy", "MAE", "MSE", "MAPE", "D-ratio"]
    corr_mat = pd.DataFrame(np.array(corr_mats).mean(axis=0), index=indices, columns=cols)
    sn.heatmap(corr_mat, annot=True, cmap=blue_cmap)
    plt.title("Correlation between error- and investment-metrics", fontsize=26, pad=30)
    plt.show()
