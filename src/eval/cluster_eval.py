# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：cluster_eval.py
@IDE ：PyCharm
"""
from definitions import *
from sklearn import metrics


def cluster_evaluate(model, X):
    labels = model.labels_
    res_dict = {
        'Silhouette coefficient': metrics.silhouette_score(X, labels),
        'Calinski-Harabasz Index': metrics.calinski_harabasz_score(X, labels),
        'Davies-Bouldin Index': metrics.davies_bouldin_score(X, labels)
    }
    return res_dict


def print_clustering_results(evaluation_results):
    model_names = list(evaluation_results[0].keys())
    metrics = ['Silhouette coefficient', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    result_dataframe = pd.DataFrame(columns=['Model'] + metrics)
    result_dataframe['Model'] = model_names
    for i, model_name in enumerate(model_names):
        for j, metric in enumerate(metrics):
            metric_values = [evaluation_results[idx_][model_name][metric] for idx_ in
                             range(len(evaluation_results))]
            result_dataframe.iloc[i, j + 1] = '{:.4f}±{:.4f}'.format(
                np.mean(metric_values), np.std(metric_values))
    print(result_dataframe.to_markdown(index=False))
    result_dataframe.to_excel(TABLE_DIR + 'cluster_res.xlsx', index=False)
