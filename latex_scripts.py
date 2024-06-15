import pandas as pd


def metric_csv_to_latex_table(filename, dataset, model):
    metrics = pd.read_csv(filename)
    metrics = metrics[
        metrics["path"].str.split("/").str.get(3).str.split("_").str.get(1) == dataset
    ]
    metrics = metrics[metrics["type"] == model]
    metrics["method"] = (
        metrics["path"].str.split("/").str.get(3).str.split("_").str.get(2)
    )
    latex_string = ""

    acc_metrics = [
        "accuracy",
        "pos_f1",
        "pos_recall",
        "pos_precision",
        "neg_f1",
        "neg_recall",
        "neg_precision",
    ]
    inv_metrics = [
        "long_short5",
        "long_short10",
        "long_short20",
        "long_short50",
        "sharpe5",
        "sharpe10",
        "sharpe20",
        "sharpe50",
        "d-ratio5",
        "d-ratio10",
        "d-ratio20",
        "d-ratio50"
    ]

    drawdown_metrics = [
        "max_drawdown5",
        "max_drawdown10",
        "max_drawdown20",
        "max_drawdown50",
        "longest_drawdown_period5",
        "longest_drawdown_period10",
        "longest_drawdown_period20",
        "longest_drawdown_period50"
    ]

    error_metrics = [
        "mse",
        "mae",
        "mape"
    ]

    cols = acc_metrics + inv_metrics + drawdown_metrics + error_metrics

    top_three = {}

    for metric in cols:
        if metric in error_metrics:
            top_three[metric] = metrics[metric].sort_values().iloc[2]
        else:
            top_three[metric] = metrics[metric].sort_values().iloc[-3]

    for j, column in enumerate(cols):
        latex_string += f'{column.replace("max_drawdown", "MDD ").replace("longest_drawdown_period", "LDD ").replace("sharpe", "Sharpe ").replace("d-ratio", "D-ratio ").replace("accuracy", "Accuracy").replace("pos_f1", "Positive F1").replace("pos_recall", "Positive Recall").replace("pos_precision", "Positive Precision").replace("neg_f1", "Negative F1").replace("neg_recall", "Negative Recall").replace("neg_precision", "Negative Precision").replace("long_short", "Return ")}'
        for i in metrics.index.values:
            if column in ["mse", "mae", "mape"]:
                val = round(metrics[column][i], 4)
                if val <= top_three[column]:
                    latex_string += f" & \\textbf{{{val}}}"
                else:
                    latex_string += f" & {val}"
            else:
                val = round(metrics[column][i], 3)
                if val >= top_three[column]:
                    latex_string += f" & \\textbf{{{val}}}"
                else:
                    latex_string += f" & {val}"
        latex_string += "\\\\ \\hline\n"
    print(latex_string)