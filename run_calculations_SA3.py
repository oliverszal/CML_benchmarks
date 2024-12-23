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

run_name = 'SA_large_range_80%'

file_name = 'Polaris/' + data_name + '.parquet'
matrix, matrix_labels, names = svm.read_parquet(file_name)

slice_sizes = [int(177/n) for n in [4,3,2,1]]
# limits = [34,26,18,12]
N = 10

# with open('results/slice_study_pgp-broccatelli_QA.pkl', 'rb') as file:
#     slice_study_results = pickle.load(file)
# start = len(slice_study_results['param_grids'])
# data_sets, test_values_list, best_params, param_tables, cv_results, search_spaces, = slice_study_results['datasets'], slice_study_results['test_values'], slice_study_results['best_params'], slice_study_results['param_tables'], slice_study_results['cv_results'], slice_study_results['param_grids']

start = 0
test_values_list, best_params, param_tables, cv_results = ({slice_size: [] for slice_size in slice_sizes} for _ in range(4))
datasets, search_spaces = [], []

for i in tqdm(range(start,N)):
    vectors, labels, test_vectors, test_labels = svm.prepare_data_sets(matrix, matrix_labels, train_percentage=80, positive_negative_ratio=None, max_train_size=None, min_train_size=None, seed=i, normalize_data=True, print_info=True)
    default_gamma = 1 / (vectors.shape[1] * vectors.var())
    search_space = {
        'base': sks.Categorical([2,10,100]),
        'num_encoding': sks.Integer(1,4),
        'kernel': sks.Real(0.05 * default_gamma, 10 * default_gamma, prior='uniform'),
        'penalty': sks.Real(0,10),
    }
    search_spaces += [search_space]
    datasets += [(vectors, labels, test_vectors, test_labels)]
    inner_estimator = svm.qSVM_estimator(solver='SA', adjust_bias=False)
    for s, slice_size in enumerate(slice_sizes):
        print('Slice size: ', slice_size)
        estimator = svm.slice_estimator(estimator=inner_estimator, slice_size=slice_size, force_unbiased=False, adjust_outer_bias=True, seed=0)
        # limit = limits[s]
        f, param_table, opt, optimal_params = svm.hyperparameter_optimization(estimator=estimator, search_space=search_space, vectors=vectors, labels=labels, folds=4, mode='bayes', limit=None, filter=True, print_info='q', seed=0)
        # save results:
        test_values = f(test_vectors)
        test_values_list[slice_size] += [test_values]
        param_tables[slice_size] += [param_table]
        best_params[slice_size] += [optimal_params]
        cv_results[slice_size] += [opt.cv_results_]    
        slice_study_results = {
            'data_sets' : datasets,
            'test_values': test_values_list, 
            'best_params': best_params, 
            'param_tables': param_tables, 
            'cv_results': cv_results, 
            'param_grids': search_spaces
        }
        with open('results/slice_study_' + data_name + '_' + run_name + '.pkl','wb') as f:
            pickle.dump(slice_study_results, f)
        # skopt.dump(opt, 'results/opts/' + data_name + '_' + run_name + '_N' + str(i) + '_slice_size' + str(slice_size) + '.pkl', store_objective=True)