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
from matplotlib.backends.backend_pdf import PdfPages


from datasets import *
from metrics import *

import myclustering


def main(args):
    benchmark(args,
        datasets=[
            { 'name' : 'iris',
              'load_function' : load_iris,
              'class' : 'species',},
            { 'name' : 'adult',
              'load_function' : load_adult,
              'class' : 'Target',
              'subset' : 1000},
            { 'name' : 'ctg',
              'load_function' : load_ctg,
              'class' : 'CLASS',
              'subset' : 1000,
            }
        ],
        metrics=[
            { 'name' : 'Adjusted Rand Score',
              'function' : adjusted_rand_score, },
            { 'name' : 'Purity',
              'function' : purity, },
        ],
        algorithms=[
            # { 'name' : 'sklearn KMeans',
            #   'function' : sklearn.cluster.KMeans,
            #   'args' : {'max_iter' : 100, 'n_init' : 1},
            #   'rep_count' : 10},
            { 'name' : 'MyKMeans',
              'function' : myclustering.MyKMeans,
              'args' : {'max_iter' : 100, 'initialization_type' : 'Forgy'},
              'rep_count' : 20},
            { 'name' : 'MyKMeans',
              'function' : myclustering.MyKMeans,
              'args' : {'max_iter' : 100, 'initialization_type' : 'MeanStd'},
              'rep_count' : 20},
            # { 'name' : 'sklearn AgglomerativeClustering',
            #   'function' : sklearn.cluster.AgglomerativeClustering,
            #   'args' : {'linkage':'complete'},
            #   'rep_count' : 1},
            { 'name' : 'MyHAGClustering',
              'function' : myclustering.MyHAGClustering,
              'args' : {'use_heap': True },
              'rep_count' : 1},
        ],
        scaling_methods=[
            # { 'name' : 'None',
            #   'function' : lambda x: x, },
            { 'name' : 'min-max',
              'function' : lambda df: (df-df.min(axis=0))/(df.max(axis=0)-df.min(axis=0)) },
            { 'name' : 'mean-std',
              'function' : lambda df: (df-df.mean(axis=0))/df.std(axis=0) },
        ],
        n_clusterss=list(range(1,15))
    )


def convert_to_numeric(data, class_name):
    """
    Converts nominal attributes (different than the class) into numeric,
    by splitting them into multiple binary attributes (if std of binary column is > 0).

    Parameters
    ----------
    data : dataframe
        dataset to convert
    """
    print('Initial shape:', data.shape)
    print('Initial columns:', data.columns)
    print(data.head(10))
    new_data = pd.DataFrame()
    for col_name in data.columns:
        values = data[col_name].unique()
        if not isinstance(values[0], str) or col_name == class_name:
            new_data[col_name] = data[col_name]
        else:
            for val in values:
                new_col = data[col_name] == val
                if np.std(new_col) > 0:
                    new_data[col_name + '_' + str(val)] = new_col
    print('New shape:', new_data.shape)
    print('New columns:', new_data.columns)
    print(new_data.head(10))
    return new_data


def get_subset(data, count, random_seed=123):
    """ Returns dataframe with randomly sampled rows.
    """
    import random
    rand = random.Random(random_seed)
    indices = rand.sample(range(data.shape[0]), count)
    return data.loc[indices]


def get_dataset_info_str(metadata, data):
    ret = ''
    counts = str(data.groupby(metadata['class']).size()).split()
    ret += f'Class: {counts[0]}, Data shape: {data.shape}, values samples count (label, count):\n'
    counts = counts[1:-2]
    ret += '  '.join(map(str,zip(counts[::2], counts[1::2])))
    return ret


def benchmark(args, datasets, metrics, algorithms, scaling_methods, n_clusterss):
    """
    Generic benchmark function.

    Parameters
    ----------
    args : list
        command line parameters concerning output of results and plot drawing

    datasets : list
        list of dicts with keys: name, load_function, class

    metrics : list
        list of dicts with keys: name, function

    algorithms : list
        list of dicts with keys: name, function, args, rep_count

    scaling_methods : list
        list of dicts with keys: name, function

    n_clusterss : list
        list of integers -- n_clusters values to perform clustering experiments with
    """

    font = {'family' : 'DejaVu Sans',
            'size'   : 18}
    matplotlib.rc('font', **font)
    figures = []

    for dataset in datasets:
        print('========================== Dataset:', dataset['name'])
        data = dataset['load_function']()
        if 'subset' in dataset:
            data = get_subset(data, dataset['subset'])
        data = convert_to_numeric(data, class_name=dataset['class'])
        labels_true = data[dataset['class']]
        data_no_class = data[[c for c in data.columns if c!=dataset['class']]]
        dataset_info = get_dataset_info_str(dataset, data)
        assert len(data_no_class.columns) == len(data.columns) - 1
        data_no_class = np.array(data_no_class.values, dtype=np.float64)
        print('-----Experiments')

        for metric in metrics:
            figures.append(plt.figure(figsize=(args.plot_width/args.plot_dpi,
                                args.plot_height/args.plot_dpi), dpi=args.plot_dpi))
            plot_lines = []

            for algorithm in algorithms:
                for scaling_method in scaling_methods:
                    scores = []
                    plot_label = 'alg={}({}); scaling={}; rep_count={}'.format(
                            algorithm['name'],
                            ', '.join([f'{k}={v}' for k,v in algorithm['args'].items()]),
                            scaling_method['name'], algorithm['rep_count'])
                    print(plot_label)
                    scaled_data = scaling_method['function'](data_no_class)

                    for n_clusters in n_clusterss:
                        single_scores = []
                        for _ in range(algorithm['rep_count']):
                            alg_object = algorithm['function'](n_clusters=n_clusters, **algorithm['args'])
                            alg_object.fit(scaled_data)
                            labels_pred = alg_object.labels_
                            assert labels_pred.shape == (data.shape[0],) # TODO: remove
                            assert (0 <= labels_pred).all() # TODO: remove
                            assert (labels_pred < n_clusters).all() # TODO: remove
                            single_scores.append(metric['function'](labels_true=labels_true, labels_pred=labels_pred))
                        scores.append(np.mean(single_scores))
                        #if n_clusters == len(data[dataset['class']].unique()):
                        #    print('Confusion matrix for n_clusters == different class values count:')
                        #    print(pd.crosstab(labels_true, labels_pred))
                    plot_lines.append(plt.plot(n_clusterss, scores, label=plot_label)[0])

            plt.title(f'dataset={dataset["name"]}; metric={metric["name"]}' + '\n' + dataset_info)
            ax = plt.gca()
            ax.set_ylabel(metric["name"])
            ax.set_xlabel('Clusters count')
            plt.legend(handler_map={line: HandlerLine2D() for line in plot_lines})
            plt.grid(True)
            if args.show_plots:
                plt.show()

    pp = PdfPages(args.out_path)
    for fig in figures:
        pp.savefig(fig)
    pp.close()


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
