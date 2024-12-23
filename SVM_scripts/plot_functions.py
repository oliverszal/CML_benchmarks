from . import SVM_functions as svm
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as scs
from collections import OrderedDict
import pickle

__all__ = [
    'load_slice_study_results', 
    'plot_slice_study_kappas', 
    'plot_best_params', 
    'plot_kappas_per_params',
    'plot_kappa_diffs', 
    'combine_slice_study'
]

def compute_metrics_from_results(test_value_results, metric, data_sets):
    evaluated_metrics = {slice_size: [metric(test_value, data_sets[i][3]) if test_value is not None else None for i, test_value in enumerate(test_values)] for slice_size, test_values in test_value_results.items()}
    return evaluated_metrics

def load_slice_study_results(results_file, data_file=None, complete_file=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_file = os.path.abspath(os.path.join(script_dir, '..', 'results', results_file))
    with open(result_file, 'rb') as file:
        slice_study_results = pickle.load(file)
    needed_keys = ['kappas', 'further_kappas', 'other_metrics']
    needed_bools = [not key in slice_study_results for key in needed_keys]
    if np.any(needed_bools): # need to load data_sets
        if data_file:
            matrix, matrix_labels, names = svm.read_parquet(data_file)
            data_sets = [svm.prepare_data_sets(matrix, matrix_labels,**data_info) for data_info in slice_study_results['data_info']]
        elif 'data_sets' in slice_study_results:
            data_sets = slice_study_results['data_sets']
        else:
            try:
                data_file = os.path.abspath(os.path.join(script_dir, 'Polaris', results_file.split('slice_study_results_')[1].split('_')[0] + '.parquet'))
                matrix, matrix_labels, names = svm.read_parquet(data_file)
                data_sets = [svm.prepare_data_sets(matrix, matrix_labels,**data_info) for data_info in slice_study_results['data_info']]
            except FileNotFoundError:
                raise Exception('Data sets needed but not provided or found')
    extracted_info = [{}]*3 # kappas, further_kappas, other_metrics
    for i, (key, boolean) in enumerate(zip(needed_keys, needed_bools)):
        if boolean:
            if key == 'kappas':
                extracted_info[i] = compute_metrics_from_results(slice_study_results['test_values'], svm.kappa, data_sets)
            elif key == 'further_kappas':
                extracted_info[i] = {further_key: compute_metrics_from_results(results, svm.kappa, data_sets) for further_key, results in slice_study_results['further_test_values'].items()}
            elif key == 'other_metrics':
                for metric_name, metric in [('accuracy', svm.accuracy), ('AUROC', svm.AUROC), ('AUPRC',svm.AUPRC)]:
                    extracted_info[i][metric_name] = compute_metrics_from_results(slice_study_results['test_values'], metric, data_sets)
            if complete_file:
                slice_study_results[key] = extracted_info[i]
        else:
            extracted_info[i] = slice_study_results[key]

    kappas, further_kappas, other_metrics = extracted_info
    best_params = slice_study_results['best_params']
    param_tables = slice_study_results['param_tables']
    return slice_study_results, kappas, best_params, param_tables, further_kappas, other_metrics

def plot_slice_study_kappas(kappas, title='Kappas', plot_stds=False, limit=None, y_limit=None, save=None):
    #slice_sizes = kappas.keys()
    plt.violinplot(
        dataset=[[kappa for kappa in slice_kappas[:limit] if kappa is not None] for slice_size, slice_kappas in kappas.items() if len(slice_kappas) > 0],
        showmeans=True,
        bw_method=0.2,
    )
    if plot_stds:
        for s, (slice_size, slice_kappas) in enumerate(kappas.items()):
            data = [kappa for kappa in slice_kappas[:limit] if kappa is not None]
            plt.scatter([s + 1] * 2, [np.mean(data) + np.std(data), np.mean(data) - np.std(data)], marker='_', color="#1f77b4")
    # plot failed fits:
    None_count = [np.count_nonzero([kappa is None for kappa in slice_kappas[:limit]]) for slice_size, slice_kappas in kappas.items()]
    for i, count in enumerate(None_count):
        if count > 0:
            plt.scatter([i + 1 + np.ceil(n/2) * 0.05 * (-1)**n for n in range(count)], [0] * count, marker='x', color='red')
    plt.xticks(range(1,len(kappas)+1), [str(slice_size) for slice_size in kappas.keys()])
    plt.ylim(y_limit)
    plt.title(title)
    plt.grid()
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png', dpi=300)
    plt.show()

def plot_best_params(best_params):
    for param_name in list(best_params.values())[0][0].keys():
        plt.violinplot(
        dataset=[[dic[param_name] for dic in params if dic is not None] for slice_size, params in best_params.items() if params is not None and len(params) > 0],
            showmeans=True,
            bw_method=0.1,
        )
        plt.xticks(range(1,len(best_params)+1), [str(slice_size) for slice_size in best_params.keys()])
        ax = plt.gca()
        plt.title('best ' + param_name)
        plt.grid()
        plt.show()

def plot_kappa_diffs(kappas, further_kappas, title='', limit=None, y_limit=None, save=None):
    # colors = ["#1f77b4", '#ff7f0e', '#2ca02c']
    colors = ['#005478', '#FBB906', '#BCCF0F', '#BA0051', '#47BCCD', '#DAD0B9']

    def Nonedifference(a,b):
        return [a_i - b_i if a_i is not None and b_i is not None else None for a_i, b_i in zip(a, b)]

    qkappas2000 = further_kappas['more_shots']
    qkappas50 = further_kappas['fewer_shots']
    qkappas250 = kappas
    qlimited_diffs_high = {slice_size: Nonedifference(kappa, qkappas250[slice_size])[:limit] for slice_size, kappa in qkappas2000.items()}
    qlimited_diffs_low = {slice_size: Nonedifference(kappa, qkappas250[slice_size])[:limit] for slice_size, kappa in qkappas50.items()}
    violin_data_high = [[diff for diff in kappa_diff if diff is not None] for slice_size, kappa_diff in qlimited_diffs_high.items()]
    violin_data_low = [[diff for diff in kappa_diff if diff is not None] for slice_size, kappa_diff in qlimited_diffs_low.items()]
    xaxis = np.arange(0, 4*len(violin_data_high), 4)
    violin_parts_high = plt.violinplot(dataset=violin_data_high, positions=xaxis + 0.5, widths=1, showmeans=True, bw_method=0.2)
    violin_parts_low = plt.violinplot(dataset=violin_data_low, positions=xaxis - 0.5, widths=1, showmeans=True, bw_method=0.2)
    for c, violin_parts in zip(colors[:2], [violin_parts_high, violin_parts_low]):
        for pc in violin_parts['bodies']:
            pc.set_facecolor(c)
        for violin_key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            violin_parts[violin_key].set_color(c)
    # plot Nones:
    None_count_high = [np.count_nonzero([diff is None for diff in kappa_diff]) for slice_size, kappa_diff in sorted(qlimited_diffs_high.items())]
    None_count_low = [np.count_nonzero([diff is None for diff in kappa_diff]) for slice_size, kappa_diff in sorted(qlimited_diffs_low.items())]
    for j, None_count in enumerate([None_count_high, None_count_low]):
        for k, count in enumerate(None_count):
            if count > 0:
                plt.scatter([xaxis[k] + 0.5* (-1)**j + np.ceil(n/2) * 0.2 * (-1)**n for n in range(count)], [-0.15] * count, marker='x', color=colors[j])
    for slice_size, kappa_diff in qlimited_diffs_high.items():
        None_count_high = np.count_nonzero(kappa_diff is None)
        None_count_low = np.count_nonzero(qlimited_diffs_high[slice_size] is None)

    plt.legend(handles = [mpatches.Patch(color=colors[i]) for i in range(2)], labels=[r'$\kappa_{2000} - \kappa_{250}$', r'$\kappa_{50} - \kappa_{250}$'])
    plt.xticks(xaxis, [str(slice_size) for slice_size in qlimited_diffs_high.keys()])
    plt.title(title)
    plt.xlabel('Slice size')
    plt.ylabel('Difference in Kappa value')
    plt.grid()
    plt.ylim(y_limit)
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png', dpi=300)
    plt.show()

def plot_kappas_per_params(slice_study_results, limit=None, n_rel_digits=3, plot_markers=False, reference=None):
    def auto_round(arr, n_rel_digits=3):
        arr_range = np.max(arr) - np.min(arr)
        if arr_range == 0:
            return np.round(arr, n_rel_digits)  # Default to 3 decimals if all values are the same
        order_of_magnitude = np.floor(np.log10(arr_range))
        decimal_places = int(max(n_rel_digits, -order_of_magnitude + n_rel_digits - 1))
        return np.round(arr, decimal_places)
    try:
        N = len(slice_study_results['param_grids'])
    except KeyError:
        N = len(list(slice_study_results['test_values'].values())[0])
    slice_sizes = list(slice_study_results['param_tables'].keys())
    params = [key for key in list(slice_study_results['cv_results'].values())[0][0].keys() if key[:6] == 'param_']
    folds = [key for key in list(slice_study_results['cv_results'].values())[0][0].keys() if key[:5] == 'split' and key[-len('_test_kappa'):] == '_test_kappa']
    for param in params:
        kappas = {slice_size:{} for slice_size in slice_sizes}
        mean_kappas = {slice_size:{} for slice_size in slice_sizes}
        mean_kappa_stds = {slice_size:{} for slice_size in slice_sizes}
        for slice_size in slice_sizes:
            for i in range(N)[:limit]:
                if slice_study_results['cv_results'][slice_size][i] is not None:
                    for j, param_value in enumerate(auto_round(slice_study_results['cv_results'][slice_size][i][param], n_rel_digits=n_rel_digits)):
                        kappa_test_values = [slice_study_results['cv_results'][slice_size][i][fold][j] for fold in folds]
                        if param_value in kappas[slice_size]:
                            kappas[slice_size][param_value] += [value for value in kappa_test_values if not (np.isnan(value) or value is None)]
                        else:
                            kappas[slice_size][param_value] = [value for value in kappa_test_values if not (np.isnan(value) or value is None)]
                    # calculate means and stds:
                    for param_value in kappas[slice_size].keys():
                        mean_kappas[slice_size][param_value] = np.mean(kappas[slice_size][param_value])
                        mean_kappa_stds[slice_size][param_value] = np.std(kappas[slice_size][param_value])
        # plot results:
        fig, ax = plt.subplots(len(slice_sizes), figsize=(10, 1 + 1.25*len(slice_sizes)))
        if len(slice_sizes) == 1:
            ax = [ax]
        fig.tight_layout()
        for s, slice_size in enumerate(slice_sizes):
            x = mean_kappas[slice_size].keys()
            y = mean_kappas[slice_size].values()
            y_std = mean_kappa_stds[slice_size].values()
            x, y, y_std = map(np.array, zip(*sorted(zip(x,y, y_std)))) # sort values
            ax[s].plot(x, y, color="#1f78b4")
            ax[s].fill_between(x, y - y_std, y + y_std, alpha=0.3)
            ax[s].set_ylabel('Kappa')
            ax[s].set_title('slice_size ' + str(slice_size))
            ax[s].grid()
            if plot_markers:
                ax[s].scatter(x, y, marker='x', c="#1f78b4")
            # plot extra data in plot:
            if reference:
                ax[s].plot(reference[param][slice_size]['x'], reference[param][slice_size]['y'], c='red')
                ax[s].fill_between(reference[param][slice_size]['x'], reference[param][slice_size]['y'] - reference[param][slice_size]['y_std'], reference[param][slice_size]['y'] + reference[param][slice_size]['y_std'], alpha=0.3, color='red')
        ax[-1].set_xlabel(param.split('param_')[-1].split('estimator__')[-1])
        plt.show()

####### function to transform data from the old format to the new: This may be deleted at some point
def combine_slice_study(slice_study_results):
    # This needs to be tidied up and finalized
    # also have as outputs: combined_kappas, combined_best_params, combined_param_tables, combined_further_kappas, combined_other_metrics
    cv_results = slice_study_results['cv_results']
    N = len(list(cv_results.values())[0])
    slice_sizes = list(cv_results.keys())
    keys = list(list(cv_results.values())[0][0].keys())
    cummulative_keys = [key for key in keys if key[-5:] == '_time' or key[:6] == 'param_' or key[:5] == 'split' or key[:5] == 'mean_' or key[:4] == 'std_']
    # 'rank_test_accuracy', 'rank_test_kappa', 'rank_train_kappa', 'rank_train_accuracy'
    combined_cv_results = [{} for _ in range(N)]
    combined_best_params, combined_best_scores, combined_param_tables  = [[] for _ in range(3)]
    refit_metric = 'kappa'
    for i in range(N):
        for key in keys:
            if key in cummulative_keys:
                combined_cv_results[i][key] = np.reshape([cv_results[slice_size][i][key] for slice_size in slice_sizes], -1)
            elif key == 'params':
                combined_cv_results[i][key] = np.reshape([[OrderedDict(partial_params, param_slice_size=slice_size) for partial_params in cv_results[slice_size][i][key]]  for slice_size in slice_sizes], -1)
            elif key[:5] == 'rank_':
                sort_reference_key = key.replace('rank', 'mean')
                combined_cv_results[i][key] = scs.rankdata(- combined_cv_results[i][sort_reference_key], method='max')
        combined_cv_results[i]['param_slice_size'] = np.reshape([slice_size * np.ones(len(cv_results[slice_size][i][keys[0]])) for slice_size in slice_sizes], -1)
        # best params:
        best_index = combined_cv_results[i][f"rank_test_{refit_metric}"].argmin()
        combined_best_params += [combined_cv_results[i]["params"][best_index]]
        combined_best_scores += [combined_cv_results[i][f"mean_test_{refit_metric}"][best_index]]
        # param tables:
        combined_param_tables += [svm.cv_results_to_df(combined_cv_results[i])]

        kappas = []
    return combined_cv_results, kappas, combined_best_params, combined_param_tables, combined_best_scores

def transform_data(results_file):
    slice_study_results, kappas, best_params, param_tables, further_kappas, other_metrics = svm.load_slice_study_results(None, results_file)
    # save further_kappas into file
    slice_study_results['kappas'] = kappas
    # put further test_values into further_test_values
    # put further kappas into further_kappas
    further_test_values_keys = [key for key in slice_study_results.keys() if len(key.split('test_values')) > 1 and not key in ['test_values', 'all_test_values']]
    slice_study_results['further_test_values'] = {}
    slice_study_results['further_kappas'] = {}
    for key in further_test_values_keys :
        slice_study_results['further_test_values'][key.split('test_values_')[1]] = slice_study_results[key]
        del slice_study_results[key]
        kappa_key = key.replace('test_values', 'kappas')
        slice_study_results['further_kappas'][kappa_key.split('kappas_')[1]] = further_kappas[kappa_key]
    # save other_metrics into file
    slice_study_results['other_metrics'] = other_metrics
    # include data_info
    data_info = [dict(train_percentage=50, positive_negative_ratio=None, max_train_size=None, min_train_size=None, seed=i, normalize_data=True, print_info=True) for i in range(len(slice_study_results['data_sets']))]
    slice_study_results['data_info'] = data_info
    # include shots info
    data_names = ['ncats-solubility', 'bbb-martins', 'pgp-broccatelli', 'broccatelli_unbiased']
    shots_250_list = np.reshape([[data_name + '_QA', data_name + '_SA', data_name + '_SA_limited'] for data_name in data_names], -1)
    shots_200_list = [data_name + 'QA_grid' for data_name in data_names]
    if len(results_file.split('classical')) <= 1:
        slice_study_results['shots_info'] = {}
        if results_file.split('results/slice_study_')[1].split('.pkl')[0] in shots_250_list:
            slice_study_results['shots_info']['normal'] = 250
        elif results_file.split('results/slice_study_')[1].split('.pkl')[0] in shots_200_list:
            slice_study_results['shots_info']['normal'] = 200
        if 'more_shots' in slice_study_results['further_test_values']:
            slice_study_results['shots_info']['more_shots'] = 2000
        if 'fewer_shots' in slice_study_results['further_test_values']:
            slice_study_results['shots_info']['fewer_shots'] = 50
        if 'median_params' in slice_study_results['further_test_values']:
            slice_study_results['shots_info']['median_params'] = 50
    # delete data_sets
    del slice_study_results['data_sets']
    # write files
    with open(results_file, 'wb') as f:
        pickle.dump(slice_study_results, f)