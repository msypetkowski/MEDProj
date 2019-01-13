#!/usr/bin/env python3

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



def main(args):
    benchmark(args,
        datasets=[
            { 'name' : 'iris',
              'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
              'class' : 'species', },
        ],
        metrics=[
            { 'name' : 'adjusted_rand_score',
              'function' : adjusted_rand_score, },
        ],
        algorithms=[
            { 'name' : 'sklearn.cluster.KMeans',
              'function' : sklearn.cluster.KMeans, },
        ],
        scaling_methods=[
            { 'name' : 'None',
              'function' : lambda x: x, },
        ],
        n_clusterss=list(range(1,5))
    )


def benchmark(args, datasets, metrics, algorithms, scaling_methods, n_clusterss):
    """
    Generic benchmark function.

    Parameters
    ----------
    args : list
        command line parameters concerning output of results and plot drawing

    datasets : list
        list of dicts with keys: name, url, class

    metrics : list
        list of dicts with keys: name, function

    algorithms : list
        list of dicts with keys: name, function

    scaling_methods : list
        list of dicts with keys: name, function

    n_clusterss : list
        list of integers -- n_clusters values to perform experiments with
    """

    font = {'family' : 'DejaVu Sans',
            'size'   : 16}
    matplotlib.rc('font', **font)

    for dataset in datasets:
        print('---------Processing dataset:', dataset['name'])
        data = pd.read_csv(dataset['url']) # TODO: other formats
        print('Columns:', list(data.columns))
        labels_true = data[dataset['class']]
        data_no_class = data[[c for c in data.columns if c!=dataset['class']]]
        assert len(data_no_class.columns) == len(data.columns) - 1
        data_no_class = data_no_class.values

        for metric in metrics:
            plt.figure(figsize=(args.plot_width/args.plot_dpi,
                                args.plot_height/args.plot_dpi), dpi=args.plot_dpi)
            plot_lines = []

            for algorithm in algorithms:
                for scaling_method in scaling_methods:
                    scaled_data = scaling_method['function'](data_no_class)
                    scores = []
                    for n_clusters in n_clusterss:
                        alg_object = algorithm['function'](n_clusters=n_clusters)
                        alg_object.fit(data_no_class)
                        labels_pred = alg_object.labels_
                        scores.append(metric['function'](labels_true=labels_true, labels_pred=labels_pred))
                    plot_label = f'algorithm={algorithm["name"]}; scaling_method={scaling_method["name"]}'
                    plot_lines.append(plt.plot(n_clusterss, scores, label=plot_label)[0])

            plt.title(f'dataset={dataset["name"]}; metric={metric["name"]}')
            ax = plt.gca()
            ax.set_ylabel(metric["name"])
            ax.set_xlabel('n_clusters')
            plt.legend(handler_map={line: HandlerLine2D() for line in plot_lines})
            plt.grid(True)
            if args.show_plots:
                plt.show()


if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-s, --show_plots', dest='show_plots', action='store_true',
                help="Show plots duing experiments.")
        parser.add_argument('-w, --plot_width', dest='plot_width', default=1500, type=int)
        parser.add_argument('-h, --plot_height', dest='plot_height', default=1000, type=int)
        parser.add_argument('-d, --plot_dpi', dest='plot_dpi', default=80, type=int)
        parser.add_argument('-o, --out_path', dest='out_path', default='plots.pdf', type=str)
        return parser.parse_args()
    main(parse_args())
