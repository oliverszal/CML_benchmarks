import numpy as np
import SVM_scripts as svm
import sklearn as skl
import skopt
import skopt.space as sks
from tqdm import tqdm
import pickle

# data_name = 'pgp-broccatelli'
# data_name = 'bbb-martins'
data_name = 'ncats-solubility'

run_name = 'classical'

file_name = 'Polaris/' + data_name + '.parquet'
matrix, matrix_labels, names = svm.read_parquet(file_name)

slice_sizes = [36,40,50,69,123] # [int(177/n) for n in [4,3,2,1]] + [205,246,307,460,613,919,1226] # + [None]
limit = 12
# limit = 20
# limits = [34,26,18,12]
N = 20

try: 
    with open('results/slice_study_' + data_name + '_' + run_name + '.pkl', 'rb') as file:
        slice_study_results = pickle.load(file)
except FileNotFoundError:
    slice_study_results = {}

slice_keys = ['test_values', 'best_params', 'param_tables', 'cv_results']
keys = ['data_sets', 'param_grids']
for key in slice_keys + keys:
    if key not in slice_study_results:
        if key in slice_keys:
            slice_study_results[key] = {slice_size:[] for slice_size in slice_sizes}
        else:
            slice_study_results[key] = []
    elif key in slice_keys:
        for slice_size in slice_sizes:
            if slice_size not in slice_study_results[key]:
                slice_study_results[key][slice_size] = []

# start = len(slice_study_results['param_grids'])
start = 0

for i in tqdm(range(start,N)):
    vectors, labels, test_vectors, test_labels = svm.prepare_data_sets(matrix, matrix_labels, train_percentage=50, positive_negative_ratio=None, max_train_size=None, min_train_size=None, seed=i, normalize_data=True, print_info=True)
    default_gamma = 1 / (vectors.shape[1] * vectors.var())
    search_space = {
        'upper_bound': sks.Real(1,1000, prior='log-uniform'),
        'kernel': sks.Real(0.05 * default_gamma, 2 * default_gamma, prior='log-uniform'),
    }
    if i >= len(slice_study_results['param_grids']):
        slice_study_results['param_grids'] += [search_space]
    if i >= len(slice_study_results['data_sets']):
        slice_study_results['data_sets'] += [(vectors, labels, test_vectors, test_labels)]
    inner_estimator = svm.cSVM_estimator(solver='SVC', adjust_bias=False)
    for s, slice_size in enumerate(slice_sizes):
        print('Slice size: ', slice_size)
        estimator = svm.slice_estimator(estimator=inner_estimator, slice_size=slice_size, force_unbiased=bool(data_name == 'bbb-martins'), adjust_outer_bias=True, seed=0)
        try:
            f, param_table, opt, optimal_params = svm.hyperparameter_optimization(estimator=estimator, search_space=search_space, vectors=vectors, labels=labels, folds=4, mode='bayes', limit=limit, filter=None, print_info='q', seed=0)
            test_values = f(test_vectors)
            cv_results = opt.cv_results_
        except ValueError:
            print('Run failed, putting None')
            test_values, optimal_params, param_table, cv_results = [None] * 4
        slice_study_results['test_values'][slice_size] += [test_values]
        slice_study_results['best_params'][slice_size] += [optimal_params]
        slice_study_results['param_tables'][slice_size] += [param_table]
        slice_study_results['cv_results'][slice_size] += [cv_results] 
        with open('results/slice_study_' + data_name + '_' + run_name + '.pkl','wb') as f:
            pickle.dump(slice_study_results, f)
        # skopt.dump(opt, 'results/opts/' + data_name + '_' + run_name + '_N' + str(i) + '_slice_size' + str(slice_size) + '.pkl', store_objective=True)