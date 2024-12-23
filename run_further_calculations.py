import numpy as np
import SVM_scripts as svm
import sklearn as skl
from tqdm import tqdm
import pickle

def shave_param_names(param_dict):
    dict_copy = param_dict.copy()
    for param_name, value in param_dict.items():
        prefix = 'sliced__estimator__'
        if len(param_name.split(prefix)) > 1:
            shaved_param_name = param_name.split(prefix)[1]
            dict_copy[shaved_param_name] = dict_copy.pop(param_name)
    return dict_copy

# data_name = 'pgp-broccatelli'
data_name = 'bbb-martins'
# data_name = 'ncats-solubility'

run_name = 'SA'

results_file = 'results/slice_study_' + data_name + '_' + run_name + '.pkl'
data_file_name = 'Polaris/' + data_name + '.parquet'
matrix, matrix_labels, names = svm.read_parquet(data_file_name)

with open(results_file, 'rb') as file:
    slice_study_results = pickle.load(file)
slice_sizes = list(slice_study_results['test_values'].keys())
# slice_sizes = [59,88,177]

N = len(list(slice_study_results['test_values'].values())[0])
# N = 18

key = 'test_values_more_shots'
chosen_params = slice_study_results['best_params']
# key = 'test_values_median_params'
param_median = {slice_size: {param: np.median([slice_study_results['best_params'][slice_size][i][param] for i in range(len(slice_study_results['best_params'][slice_size])) if slice_study_results['best_params'][slice_size][i] is not None]) for param in slice_study_results['best_params'][slice_size][0].keys()} for slice_size in slice_sizes}
# clean up (make num_encoding an integer):
for slice_size in param_median.keys():
    for param in param_median[slice_size].keys():
        if param[-len('num_encoding'):] == 'num_encoding':
            param_median[slice_size][param] = int(param_median[slice_size][param])
# print('Median of optimal params:', param_median)
# chosen_params = {slice_size:[param_median[slice_size]] * N for slice_size in slice_sizes}

slice_study_results[key] = {slice_size:[] for slice_size in slice_sizes} # if previous results do not exist yet
# start = 0
start = len(list(slice_study_results[key].values())[0])
for i in tqdm(range(start,N)):
    vectors, labels, test_vectors, test_labels = svm.prepare_data_sets(matrix, matrix_labels, train_percentage=50, positive_negative_ratio=None, max_train_size=None, min_train_size=None, seed=i, normalize_data=True, print_info=False)
    for slice_size in slice_sizes:
        print('Slice_size: ', slice_size)
        params = chosen_params[slice_size][i]
        if params is None:
            new_test_values = None
        else:    
            inner_estimator = svm.qSVM_estimator(solver=('SA', 2000), adjust_bias=False, **shave_param_names(params))
            # inner_estimator = svm.cSVM_estimator(solver='SVC', adjust_bias=False, **shave_param_names(params))
            estimator = svm.slice_estimator(estimator=inner_estimator, slice_size=slice_size, force_unbiased=bool(data_name == 'bbb-martins'), adjust_outer_bias=True, seed=0, print_info=False)
            try:
                estimator.fit(vectors, labels)
                f = estimator.decision_function
                new_test_values = f(test_vectors)
                print('Kappa: ', svm.kappa(new_test_values, test_labels))
            except Exception as e:
                print(e)
                new_test_values = None
                print('Fit failed, putting None as test_values')
        slice_study_results[key][slice_size] += [new_test_values]
        with open(results_file,'wb') as f:
            pickle.dump(slice_study_results, f)