"""
Author: Michał Sypetkowski

Comparison of KMeans and HAG clustering algoritms.
Measurements are done using various metrics, datasets, and data pre-processing methods.
"""


import pandas as pd
import numpy as np
import sklearn.cluster
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D



def main():
    experiments = {
        'dataset' : [
            {
                'name' : 'iris',
                'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
                'class' : 'species',
            }
        ],

        'metric' : [
            {
                'name' : 'adjusted_rand_score',
                'function' : adjusted_rand_score,
            }
        ],

        'algorithm' : [
            'name' : 'sklearn.cluster.KMeans',
            'function' : sklearn.cluster.KMeans,
        ]

        'scaling_method' : [
            {
                'name' : 'None',
                'function' : lambda x: x,
            },
        ],

        'n_clusters' : list(range(1,5)),
    }
    benchmark(experiments)


def benchmark(experiments):
    """
    Generic benchmark function.

    Parameters
    ----------
    experiments : dict
        Contains information such as:
        dataset, metric, algorithm, data scaling method,
        n_clusters values to measure with.
    """

    font = {'family' : 'normal',
            'size'   : 16}
    matplotlib.rc('font', **font)

    for dataset in experiments['dataset']:
        print('---------Processing dataset:', dataset['name'])
        data = pd.read_csv(dataset['url']) # TODO: other formats
        print('Columns:', data.columns)
        labels_true = data[dataset['class']]
        data_no_class = data[[c for c in data.columns if c!=dataset['class']]]
        assert len(data_no_class.columns) == len(data.columns) - 1
        data_no_class = data_no_class.values

        for metric in experiments['metric']:
            plot_dpi = 80
            plt.figure(figsize=(1500/plot_dpi, 1500/plot_dpi), dpi=plot_dpi)
            plot_lines = []

            for algorithm in experiments['algorithm']:
                for scaling_method in experiments['scaling_method']:
                    scaled_data = scaling_method['function'](data_no_class)
                    scores = []
                    for n_clusters in experiments['n_clusters']:
                        alg_object = algorithm(n_clusters=n_clusters)
                        alg_object.fit(data_no_class)
                        labels_pred = alg_object.labels_
                        scores.append(metric['function'](labels_true=labels_true, labels_pred=labels_pred))
                    plot_label = f'algorithm={str(algorithm)}, scaling_method={scaling_method["name"]}'
                    plot_lines.append(plt.plot(experiments['n_clusters'], scores, label=plot_label)[0])

            ax = plt.gca()
            ax.set_ylabel(f'dataset={dataset["name"]}, metric={metric["name"]}')
            ax.set_xlabel('n_clusters')
            plt.legend(handler_map={line: HandlerLine2D() for line in plot_lines})
            plt.grid(True)
            # plt.savefig('results.png', dpi=plot_dpi)
            plt.show()


if __name__ == '__main__':
    main()
