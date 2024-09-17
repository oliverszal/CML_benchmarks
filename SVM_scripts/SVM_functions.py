import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import rdkit
from rdkit.Chem import AllChem
import inspect
import copy
import itertools
import time
import neal
import dwave
import minorminer
import dimod
import sklearn as skl
from sklearn.gaussian_process.kernels import Kernel
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
import imblearn as imb
try:
    import pyomo.environ as pyo
    from amplpy import modules
except Exception:
    print('Pyomo solver unusable')
from dwave.system import DWaveSampler, EmbeddingComposite, AutoEmbeddingComposite, CutOffComposite, FixedEmbeddingComposite, LazyFixedEmbeddingComposite
from dwave.system import DWaveCliqueSampler
from dwave.system import LeapHybridCQMSampler
try:
    simulated_sampler = neal.SimulatedAnnealingSampler()
    import dwave_token
    # DWave_sampler = EmbeddingComposite(DWaveSampler(token=dwave_token.value))
    DWave_sampler = AutoEmbeddingComposite(DWaveSampler(token=dwave_token.value))
    clique_sampler = DWaveCliqueSampler(token=dwave_token.value)
    hybrid_sampler = LeapHybridCQMSampler(token=dwave_token.value)
    lazy_sampler = LazyFixedEmbeddingComposite(DWaveSampler(token=dwave_token.value))
except Exception as e:
    if e.__str__() == "name 'dwave_token' is not defined" or e.__str__() == "No module named 'dwave_token'":
        print('D-Wave Token unspecified. Can not choose any quantum solver')

__all__ = [
    'cSVM', 
    'qSVM', 
    'decision_function', 
    'averaged_decision_function', 
    'averaged_decision_function_from_samples', 
    'gauss_kernel', 
    'polynomial_kernel', 
    'linear_kernel', 
    'read_compounds',
    'read_parquet',
    'prepare_data_sets', 
    'slice_training_data', 
    'plot_compounds', 
    'boxplot_compounds',
    'boxplot_compounds2',
    'plot_training_data_with_decision_boundary',
    'kappa',
    'accuracy',
    'AUROC',
    'AUPRC',
    'ROC_curve',
    'PR_curve',
    'print_measures',
    'cSVM_estimator',
    'qSVM_estimator',
    'slice_estimator',
    'hyperparameter_optimization',
    'manual_classifier',
    'mean_estimator'
]

lazy_embedding_timeout = 60
def make_QUBO_upper_triangle(Q):
    Q = np.array(Q)
    return(np.triu(Q) + np.tril(Q,-1).T)

def transform_kernel_argument(kernel, solver, vectors):
    if kernel == None: # standard value
        kernel = 1 / (len(vectors[0]) * vectors.var())
    if isinstance(kernel, (int, float)): # if a number was given, use radial kernel
            kernel = ('rbf', kernel, 0, 0)
    if isinstance(kernel, tuple):
        assert kernel[0] in ['rbf', 'radial', 'gauss', 'poly', 'linear'], 'Invalid form of kernel specified'
        if solver == 'sklearn':
            if kernel[0] in ['rbf', 'radial', 'gauss']:
                kernel, gamma, degree, coef0 = 'rbf', kernel[1], 3, 0
            elif kernel[0] == 'poly':
                kernel, gamma, degree, coef0 = 'poly', *kernel[1:4]
            else:
                kernel, gamma, degree, coef0 = 'linear', 'scale', 3, 0
        else:
            if kernel[0] in ['rbf', 'radial', 'gauss']:
                kernel, gamma, degree, coef0 = gauss_kernel(kernel[1]), kernel[1], *[0]*2
            elif kernel[0] == 'poly':
                kernel, gamma, degree, coef0 = polynomial_kernel(d=kernel[2],c=kernel[3], gamma=kernel[1]), *kernel[1:4]
            else:
                kernel, gamma, degree, coef0 = linear_kernel(kernel[1]), kernel[1], *[0]*2
    elif callable(kernel):
        kernel, gamma, degree, coef0 = kernel, *[0]*3
    else:
        raise Exception('Invalid form of kernel specified')
    return(kernel, gamma, degree, coef0)

def transform_solver_argument(solver, QUBO, embedding=None, print_info:bool=False):
    if isinstance(solver, tuple):
        solver, time_limit, sampler, shots = 2*solver
    else:
        solver, sampler, time_limit, shots = *(2*tuple([solver])), 0, 100
    if isinstance(solver, str):
         solver, sampler = solver.lower(), sampler.lower()
    assert isinstance(solver, (int, float)) or solver in ['cplex', 'bonmin', 'sa', 'simulated annealing', 'simulated_annealing', 'hybrid', 'annealing', 'dwave', 'qa', 'quantum_annealing', 'sklearn', 'svc', 'clique', 'save_embedding', 'lazy'], 'Unknown solver specified'
    if solver in ['sklearn', 'svc']:
        solver = sampler = 'sklearn'
    elif solver in ['cplex', 'bonmin']:
        solver = pyo.SolverFactory(solver, tee=False)
        if time_limit > 0:
            if sampler == 'bonmin': # adapting to the specific solveroptions terminology
                solver.options['bonmin.time_limit'] = time_limit
            else:
                solver.options['timelimit'] = time_limit
    elif solver == 'hybrid':
        assert time_limit >= 5, 'To use the hybrid sampler, please specify a time limit of larger than 5 by ("hybrid", time_limit).'
        sampler = hybrid_sampler
    else:
        if solver == 'clique':
            sampler = clique_sampler
        elif solver in ['sa', 'simulated_annealing']:
            sampler = simulated_sampler
        elif isinstance(solver, (int, float)): # if sampler is a number, it is interpreted as the relative desired increase in zeros in the QUBO. 1 would mean that all non-linear terms are 0
            cutoff, initial_relative_zeros, QUBO = calculate_cut_off(QUBO, solver, max_non_zeros=35000)
            non_zeros = np.count_nonzero(QUBO - np.diag(np.diag(QUBO)))
            if print_info:
                print('Initial_relative_zeros: ', initial_relative_zeros)
                print('cutoff: ', cutoff)
                print('Non-zero connections: ', non_zeros)
            if embedding == 'save' or type(embedding) == dwave.embedding.transforms.EmbeddedStructure:
                sampler = lazy_sampler
                sampler.find_embedding = lambda S, T: minorminer.find_embedding(S, T, timeout=lazy_embedding_timeout)
            else:
                sampler = CutOffComposite(DWave_sampler, cutoff)
        elif embedding == 'save' or solver == 'lazy' or solver == 'save_embedding' or type(embedding) == dwave.embedding.transforms.EmbeddedStructure:
            sampler = lazy_sampler
            sampler.find_embedding = lambda S, T: minorminer.find_embedding(S, T, timeout=lazy_embedding_timeout)
        else:
            sampler = DWave_sampler
        if type(embedding) == dwave.embedding.transforms.EmbeddedStructure:
            sampler._fix_embedding(embedding)
    return(solver, sampler, time_limit, shots, QUBO)

def cSVM(vectors, labels, upper_bound, kernel=None, solver=('CPLEX',0), adjust_bias=False, print_info:bool=False):
    """Classically trains a SVM by solving the corresponding quadratic optimization problem.
    Arguments:
            vectors: array of data vectors, of shape (N,d) with d>1.
             labels: list or array of binary (1,-1) labels of shape (N).
        upper_bound: upper bound of the (positive) training coefficients. Should not be too little but does not have much impact on the results.
             kernel: information specifying the kernel used to transform the data into one dimension. The options are
                             None: radial kernel with standard gamma coefficient value of 1 / (d * vectors.var()). This option is the default.
                            float: gamma coefficient to be used with radial kernel.
                            tuple: of string in ['rbf', 'radial', 'gauss', 'poly', 'linear'] (the first 3 being the same) and corresponding coefficients, these can be viewed as the arguments of the functions 'gauss_kernel', 'polynomial_kernel' and 'linear_kernel'. 
                callable function: a function mapping arrays of vectors (of shapes (N,d)) onto an array of shape (N,N). This (n,m)-th element corresponds to the kernel acting on the n-th and m-th vectors.
             solver: information specifying the method of training (optimization). The options are
                string: 'CPLEX' or 'Bonmin' for Pyomo solvers (works only if installed). 'sklearn' or 'svc' for standard SVC algorithm by scikit learn. 'hybrid' for D-Wave hybrid sampler (This does not work for now, unless D-Wave changes something). 
                tuple: of string as above and computation time limit, except for when using scikit learn.
            samples: number of samples of solutions (classifiers) to be returned. For classical methods the number will always be one.
        adjust_bias: Boolean wether to adjust bias in a brute force search after optimization, or string. String means True but with a specified metric ('kappa' for kappa, else accuracy is used). Does not work with the scikit learn solver. Generally it is not recommended, as it is unlikely to improve on the results.
         print_info: Boolean wether info should be printed.

    Returns:
         objective: optimization objective. The lower the better. None if scikit learn solver is used.
             alpha: optimized variable values. None if scikit learn solver is used.
                 f: classifying function. If a proper kernel was specified, it can be used on an array of vectors to return an array of classifications.
    """
    N, d = np.shape(vectors)
    solver, sampler, time_limit, shots, _ = transform_solver_argument(solver, QUBO=np.zeros(shape=(N,N)), embedding=None, print_info=print_info)
    kernel, gamma, degree, coef0 = transform_kernel_argument(kernel, solver, vectors)
    if solver == 'hybrid': # see https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html#id2
        raise TypeError('Real-valued variables are currently not supported in quadratic interactions')
        def sampleset_to_alphas_classical(sampleset, samples):
            filtered_sampleset = sampleset.filter(lambda sample: any(np.array([sample.sample[n] not in [0,upper_bound] for n in range(N)]))) # filter sampleset by found support vectors
            alphas = np.array([list(sample.values()) for sample, in filtered_sampleset.data(fields=['sample'], sorted_by='energy')][:samples])
            objectives = [energy for energy, in filtered_sampleset.data(fields=['energy'], sorted_by='energy')][:samples]
            return(objectives, alphas)
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables('REAL', N, lower_bound=0, upper_bound=upper_bound)
        pairwise_kernel_terms = kernel(vectors,vectors)
        cqm.set_objective([(n, m, labels[n]*labels[m]*pairwise_kernel_terms[n,m]) for n in range(N) for m in range(N)] + [(n, -1) for n in range(N)])
        cqm.add_constraint_from_iterable([(n, labels[n]) for n in range(N)], sense='==', rhs=0)
        sampleset = hybrid_sampler.sample_cqm(
            cqm,
            time_limit = time_limit,
        )
        objectives, alphas = sampleset_to_alphas_classical(sampleset, samples=20)
        # if D-Wave hybrid at some point starts supporting this kind of problems, we could again specify the samples argument in the function
        fs = [decision_function(vectors, labels, upper_bound, kernel, alpha) for alpha in alphas]
        objective, alpha, f = objectives[0], alphas[0], fs[0]
    elif solver == 'sklearn':
        SVC = skl.svm.SVC(C=upper_bound, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        SVC.fit(vectors, labels)
        f = SVC.decision_function
        objective, alpha, bias = None, None, 0
    elif not sampler in ['cplex', 'bonmin']:
        raise Exception('Please specify a valid solver for solving continuous quadratic problems.')
    else:  # use pyomo solver otherwise (CPLEX, Bonmin etc.)
        model = pyo.ConcreteModel()
        model.points = pyo.RangeSet(0,N-1)
        model.dims = pyo.RangeSet(0,d-1)
        model.BinarySet = pyo.Set(initialize=[-1, 1], ordered=False)
        model.t = pyo.Param(model.points, within=model.BinarySet, initialize=labels)
        model.C = pyo.Param(within=pyo.NonNegativeReals, initialize=upper_bound)
        pairwise_kernel_terms = np.array(kernel(vectors,vectors))
        model.kernel_terms = pyo.Param(model.points, model.points, within=pyo.Reals, initialize={(n,m): pairwise_kernel_terms[n,m] for n in range(N) for m in range(N)})
        model.alpha = pyo.Var(model.points, within=pyo.NonNegativeReals, bounds=(0,model.C))
        model.balance = pyo.Constraint(expr=pyo.summation(model.t,model.alpha) == 0)
        model.obj = pyo.Objective(rule=lambda model: sum(1/2 * sum(model.alpha[n]*model.alpha[m]*model.t[n]*model.t[m]*model.kernel_terms[n,m] for m in model.points) - model.alpha[n] for n in model.points))
        instance = model.create_instance()
        results = solver.solve(instance)
        objective = pyo.value(instance.obj)
        alpha = np.array([pyo.value(instance.alpha[n]) for n in instance.points])
        f = decision_function(vectors, labels, upper_bound, kernel, alpha)
        bias = standard_bias(vectors, labels, upper_bound, kernel, alpha)[0]
    if adjust_bias:
        bias = optimize_bias([f], vectors, labels, upper_bound, kernel, alpha, adjust_bias)[0]
        f = f if alpha is None else decision_function(vectors, labels, upper_bound, kernel, alpha, bias=bias) # if alpha is None, the bias has to the decision function later (this is such that stuff works out with the estimators)
    return(objective, alpha, bias, f)

def qSVM(vectors, labels, base=2, num_encoding=3, penalty=True, kernel=None, solver='SA', samples=20, adjust_bias=False, embedding=None, print_info:bool=False):
    """Quantumly trains a SVM by converting the corresponding quadratic optimization problem into QUBO format and solving it with annealing methods.
    Arguments:
                vectors: array of data vectors, of shape (N,d) with d>1.
                 labels: list or array of binary (1,-1) labels of shape (N).
                   base: integer base B of binary encoding used. Determines the spacing between possible values of the variables. Default: 2 (regular integer spacing).
           num_encoding: integer number K of binaries used to incode a real number. Determines the amount 2^K of possible values of the variables. The upper bound of the variables will correspond to \sum_k^K B^k. Default: 3
                penalty: penalty weights for the penalty terms encoding the constraint of the classical SVM formulation. Setting penalty = False and choosing the hybrid solver, the constraint will be handelled by the solver. Standard of True corresponds to zero penalty.
                 kernel: information specifying the kernel used to transform the data into one dimension. The options are
                                 None: radial kernel with standard gamma coefficient value of 1 / (d * vectors.var()). This option is the default.
                                float: gamma coefficient to be used with radial kernel.
                                tuple: of string in ['rbf', 'radial', 'gauss', 'poly', 'linear'] (the first 3 being the same) and corresponding coefficients, these can be viewed as the arguments of the functions 'gauss_kernel', 'polynomial_kernel' and 'linear_kernel'. 
                    callable function: a function mapping arrays of vectors (of shapes (N,d)) onto an array of shape (N,N). This (n,m)-th element corresponds to the kernel acting on the n-th and m-th vectors.
                 solver: information specifying the method of training (optimization). The options are
                    string: 
                                    'SA': simulated annealing (standard).
                                'hybrid': D-Wave Leap hybrid solver, either as QUBO (penalty != False), or as a CQM (penalty = False).
                             'annealing': D-Wave sampler for quantum annealing. Will look for an embedding, which may take up to 1 minute and perhaps does not terminate.
                                'clique': Clique sampler for quantum annealing. Will immediately find an embedding if the QUBO dimension is less or equal than 177.
                                  'lazy': LazyFixedEmbeddingComposite sampler for quantum annealing. Makes it possible to specify embeddings and/or save them. Use this sampler if you want to reuse the embedding, for example if you cast the same training with a different penalty.
                        'save_embedding': LazyFixedEmbeddingComposite sampler for quantum annealing. Will return the embedding after the training.
                     float: Cutoff between 0 and 1. The relative desired increase in zeros in the QUBO. This it will ignore some terms such that the QUBO becomes more sparse. 1 would mean that all non-linear terms are 0. If an embedding is specified or to be saved, then the LazyFixedEmbeddingComposite sampler is used, otherwise the CutOffComposite sampler.
                     tuple: of string or float as above and an integer. For hybrid solver the integer describes the computation time limit (> 5). For all other samplers it is the number of shots (standard 100).
                samples: number of samples of solutions (classifiers) to be returned. Default is 20.
        choose_function: 'lowest' or 'mean', signifying the way to choose the final decision function from the set of samples. Default is 'mean'.
            adjust_bias: Boolean wether to adjust bias in a brute force search after optimization or string. String means True but with a specified metric ('kappa' for kappa, else accuracy is used). It seems to yield an improvement on sampling methods quite often. Default is False.
              embedding: object of type dwave.embedding.transforms.EmbeddedStructure, which would force the LazyFixedEmbeddingComposite sampler with the specified embedding or 'save', which will return the previously unspecified embedding used in the training as an dwave.embedding.transforms.EmbeddedStructure object. Default is None.
             print_info: Boolean wether info should be printed.

    Returns:
        objectives: list of optimization objectives (with the length samples). The lower the better.
            alphas: optimized variable values.
                f: final decision function.
                fs: list of classifying functions (with the length samples). If a proper kernel was specified, these can be used on an array of vectors to return an array of classifications.
         embedding: embedding object of type dwave.embedding.transforms.EmbeddedStructure, that was used in the optimization if such an object was specified as the argument embedding, or embedding = 'save' or if solver = 'save_embedding'. Else None.
    """
    B = base
    K = num_encoding
    upper_bound = sum(B**k for k in range(K)) # Argument 'upperbound' in cSVM corresponds to sum_{k=0}^{num_encoding-1} base^k
    N, d = np.shape(vectors)
    kernel, gamma, degree, coef0 = transform_kernel_argument(kernel, solver, vectors)
    if isinstance(penalty, (bool)) and penalty == True:
        penalty = 0
    def sampleset_to_alphas(sampleset, samples):
        filtered_sampleset = sampleset.filter(lambda sample: any(np.array([sum(sample.sample[K*n+k] for k in range(K)) not in [0,K] for n in range(N)]))) # filter sampleset by found support vectors
        if len(filtered_sampleset) == 0:
            raise Exception('No support vectors found')
        alphas = np.array([[sum(B**k * sample[K*n+k] for k in range(K)) for n in range(N)] for sample, in filtered_sampleset.data(fields=['sample'], sorted_by='energy')][:samples])
        objectives = [energy for energy, in filtered_sampleset.data(fields=['energy'], sorted_by='energy')][:samples]
        return(objectives, alphas)
        # in the following term I have added a factor 2 in front of the penalty, I think in the paper eq 13 they forgot it
    def QUBO_first_term(penalty):
        pairwise_kernel_terms = kernel(vectors,vectors)
        return(1/2 * make_QUBO_upper_triangle([[B**(k+j) * labels[n] * labels[m] * (pairwise_kernel_terms[n,m] + 2*penalty) for m in range(N) for j in range(K)] for n in range(N) for k in range(K)]))
    QUBO_second_term = - np.diag([B**k for n in range(N) for k in range(K)])
    QUBO = QUBO_first_term(penalty) + QUBO_second_term
    solver, sampler, time_limit, shots, QUBO = transform_solver_argument(solver, QUBO, embedding, print_info)
    assert not sampler in ['cplex', 'bonmin', 'sklearn'], 'Choose a QUBO solver.'
    bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO)
    if solver == 'hybrid':
        cqm = dimod.ConstrainedQuadraticModel.from_bqm(bqm)
        if isinstance(penalty, (bool)) and penalty == False:
            cqm.add_constraint_from_iterable([(K*n + k, B**k * labels[n]) for n in range(N) for k in range(K)], sense='==', rhs=0)
        sampleset = sampler.sample_cqm(
            cqm,
            time_limit = time_limit,
        )
    else:
        chain_strength = max(bqm.quadratic.values())
        sampleset = sampler.sample(
            bqm,
            num_reads=shots,
            chain_strength=chain_strength
        )
    sampleset.resolve() # to reduce output readout time
    if embedding == 'save' or solver == 'save_embedding' or type(embedding) == dwave.embedding.transforms.EmbeddedStructure:
        embedding = sampler.embedding
    else:
        embedding = None
    objectives, alphas = sampleset_to_alphas(sampleset, samples)
    if adjust_bias: # for this we assume that the kernel is allowed to take arrays in its first argument
        a = time.time()
        biases = optimize_bias(0, vectors, labels, upper_bound, kernel, alphas, adjust_bias, print_info)# here we assume that alpha is not None
        fs = [decision_function(vectors, labels, upper_bound, kernel, alphas[i], bias=biases[i]) for i in range(len(alphas))]
        if print_info:
            print('bias adjustment time: ', time.time()-a)
    else:
        fs = [decision_function(vectors, labels, upper_bound, kernel, alpha) for alpha in alphas]
        biases = standard_bias(vectors, labels, upper_bound, kernel, alphas)
    return(objectives, alphas, biases, fs, embedding)

def calculate_cut_off(QUBO, relative_zero_improvement, max_non_zeros=35000):
    # this function calculates the cutoff value of connections (quadratic values) of a QUBO, based on how large of a portion shall be set to zero. 
    # max_non_zeros serves as an absolute value of how many connections can be non zero at most
    # relative_zero_improvement adds to the portion which shall be zero
    N = (len(QUBO)**2 - len(QUBO)) / 2
    QUBO = make_QUBO_upper_triangle(QUBO)
    QUBO_connections = QUBO[np.triu_indices(len(QUBO),1)]
    non_zero_QUBO_connections = QUBO_connections[QUBO_connections != 0]
    values = sorted(set(np.round(abs(non_zero_QUBO_connections),4)))
    num_entries_smaller_than_values = np.array([np.count_nonzero(abs(non_zero_QUBO_connections) <= value) for value in values])
    forced_zeros = min(len(non_zero_QUBO_connections), max(len(non_zero_QUBO_connections) - max_non_zeros,0) + relative_zero_improvement*N)
    cutoff = next((value for i, value in enumerate(values) if num_entries_smaller_than_values[i] >= forced_zeros), 0)
    initial_relative_zeros = 1 - len(non_zero_QUBO_connections)/N
    new_QUBO = QUBO * (abs(QUBO) > cutoff)
    return(cutoff, initial_relative_zeros, new_QUBO)
    
def optimize_bias(fs, vectors, labels, upper_bound, kernel, alphas=None, metric='accuracy', print_info:bool=False):
    if alphas is None: # if no alpha is provided, the biases that this function returns is to be understood as the difference to the input functions fs
        standard_biases = np.zeros(len(fs))
    else: # if alpha is provided, the argument fs is ignored
        alphas = np.atleast_2d(alphas)
        fs = [decision_function(vectors, labels, upper_bound, kernel, alpha, bias=0) for alpha in alphas]
        standard_biases = standard_bias(vectors, labels, upper_bound, kernel, alphas)
    decision_values = [f(vectors) for f in fs]
    f_values_lists = [np.unique(decision_values[i], return_index=True) for i in range(len(fs))] # sorted lists of all function values without dublicates
    bias_filters =  [[j for j in range(len(f_values_lists[i][0])-1) if labels[f_values_lists[i][1][j+1]] == 1 and (labels[f_values_lists[i][1][j]] == -1 or j == 0)] for i in range(len(fs))]
    potential_bias_lists = [[standard_biases[i]] + [- np.mean(f_values_lists[i][0][j:j+2]) for j in bias_filters[i]] for i in range(len(fs))] # lists of possible bias choices (standard bias + half way between function values)
    if alphas is None:
        if metric == 'kappa':
            accuracy_lists = [[kappa(decision_values[i] + b, labels) for b in potential_bias_lists[i]] for i in range(len(fs))]
        else:
            accuracy_lists = [[np.sum(np.sign(decision_values[i] + b) == labels) for b in potential_bias_lists[i]] for i in range(len(fs))]
    else: # these cases could be optimized for slow kernel functions. In particular, the kernel terms kernel(vectors, vectors) should only be computed once while running the SVM
        if metric == 'kappa':
            accuracy_lists = [[kappa(decision_function(vectors, labels, upper_bound, kernel, alphas[i], bias=b)(vectors), labels) for b in potential_bias_lists[i]] for i in range(len(fs))]
        else:
            accuracy_lists = [[np.sum(np.sign(decision_function(vectors, labels, upper_bound, kernel, alphas[i], bias=b)(vectors)) == labels) for b in potential_bias_lists[i]] for i in range(len(fs))]
    best_biases = [potential_bias_lists[i][max(enumerate(accuracy_lists[i]), key=lambda x: x[1])[0]] for i in range(len(fs))]
    if print_info:
        print('Average change in bias:', np.mean(np.array(best_biases) - np.array(standard_biases)))
    return(best_biases)

def standard_bias(vectors, labels, upper_bound, kernel, alpha):
    # assumes that kernel accepts arrays in both its arguments
    # assume that alpha is a 2 array. Use np.atleast_2d() is needed.
    labels = np.array(labels)
    alpha = np.atleast_2d(alpha)
    N = len(vectors)
    C = upper_bound
    kernel_terms = kernel(vectors, vectors)
    b = np.einsum('fi,fi,fi->f', alpha, C - alpha, labels - np.einsum('ij,fj,j->fi', kernel_terms, alpha, labels)) / np.einsum('fi,fi->f', alpha, C - alpha)
    return(b)

def decision_function(vectors, labels, upper_bound, kernel, alpha, bias=None):
    '''Calculates the decision function out of the results of a SVM training.
    Arguments:
            vectors: array of data vectors used for the training, of shape (N,d) with d>1.
             labels: list or array of binary (1,-1) labels of shape (N) used for the training.
        upper_bound: upper bound of variables that were trained. Only needed if no bias is specified.
             kernel: callable kernel function that was used for the training. Assumed to accept arrays of vectors in both its argumentand return a 2-array of pairwise kernel values.
              alpha: array of variable values of shape (N) as results of the raining.
               bias: custom bias to be specified. The standard of None corresponds to the standard optimal bias.

    Returns:
                  f: classifying function that takes single or an array of vectors of length d.
    '''
    N = len(vectors)
    if bias == None: # standard bias:
        b = standard_bias(vectors, labels, upper_bound, kernel, np.atleast_2d(alpha))
    else: # custom bias:
        b = bias
    def f(vector):
        if len(np.shape(vector)) == 1:
            return(np.einsum('i,i,i', alpha, labels, kernel(np.atleast_2d(vector), vectors)[0]) + b)
        else:
            return(np.einsum('i,i,ji->j', alpha, labels, kernel(np.array(vector), vectors)) + b)
    return(f)

def averaged_decision_function_from_samples(vectors, labels, upper_bound, kernel, alphas, biases=None):
    '''Calculates the averaged decision function out of multiple results of a SVM training with the same data and parameters. This may be used when multiple samples are returned as training results.
    Arguments:
            vectors: array of data vectors used for the training, of shape (N,d) with d>1.
             labels: list or array of binary (1,-1) labels of shape (N) used for the training.
        upper_bound: upper bound of variables that were trained. Only needed if no bias is specified.
             kernel: callable kernel function that was used for the training. Assumed to accept (and return) arrays of vectors in its first argument.
             alphas: array of variable values of shape (samples,N) as results of the raining.
               biases: custom biases to be specified. The standard of None corresponds to the standard optimal bias.

    Returns:
                  f: classifying function that takes single or an array of vectors of length d.
    '''
    # the kernel is assumed to accept arrays in its first argument
    N = len(vectors)
    if biases is None: # standard bias:
        b = np.mean(standard_bias(vectors, labels, upper_bound, kernel, alphas))
    else: # custom bias:
        b = np.mean(biases)
    alpha = np.mean(alphas, axis=0)
    def f(vector):
        if len(np.shape(vector)) == 1:
            return(np.einsum('i,i,i', alpha, labels, kernel(np.atleast_2d(vector), vectors)[0]) + b)
        else:
            return(np.einsum('i,i,ji->j', alpha, labels, kernel(np.array(vector), vectors)) + b)
    return(f)

def averaged_decision_function(fs):
    '''Calculates the averaged decision function out of multiple decision functions of trained SVMs, not necessarily with the same data and parameters. This is slower than averaged_decision_function_from_samples but may be used when training happens in data slices.
    Arguments:
                 fs: list or array of decision functions accepting vectors of the same length d.

    Returns:
        averaged_f: classifying function that takes single or an array of vectors.
    '''
    def averaged_f(vector):
        average = np.mean([f(vector) for f in fs], axis=0)
        return(average)
    return(averaged_f)

class gauss_kernel(Kernel):
    '''Returns the radial kernel function exp(-gamma * ||x-y||^2). It corresponds to the options ('rbf', gamma), ('radial', gamma) and ('gauss', gamma) in the kernel arguments of cSVM and qSVM.
    Arguments:
         gamma: float, coefficient, see formular above.

    Returns:
        kernel: kernel function, to be applied to two arrays of vectors (yielding a 2-array corresponding all combinations of the input vectors).
    '''
    def __init__(self, gamma=None):
        self.gamma = gamma
    def __call__(self, X, Y=None):
        K = skl.metrics.pairwise.rbf_kernel(X,Y,gamma=self.gamma)
        return K
    def diag(self, X):
        return np.ones(X.shape[0])
    def is_stationary(self):
        return True

class polynomial_kernel(Kernel):
    '''Returns the polynomial kernel function (gamma * <x,y> + c)**d). It corresponds to the options ('poly', gamma, d, c) in the kernel arguments of cSVM and qSVM.
    Arguments:
             d: int, degree of polynomial
             c: float, coefficient, see formular above. Default is 0.
         gamma: float, coefficient, see formular above. Default is 1.

    Returns:
        kernel: kernel function, to be applied to two arrays of vectors (yielding a 2-array corresponding all combinations of the input vectors).
    '''
    def __init__(self, d=1 ,c=0, gamma=1):
        self.d = d
        self.c = c
        self.gamma = gamma
    def __call__(self, X, Y=None):
        K = lambda X,Y: skl.metrics.pairwise.polynomial_kernel(X,Y, degree=self.d, gamma=self.gamma, coef0=self.c)
        return K
    def diag(self, X):
        norm_squared = np.linalg.norm(X, axis=1)**2
        return (self.gamma * norm_squared + self.c)**self.d
    def is_stationary(self):
        return False

class linear_kernel(Kernel):
    '''Returns the linear kernel function gamma * <x,y>. It corresponds to the options ('linear', gamma) in the kernel arguments of cSVM and qSVM.
    Arguments:
         gamma: float, coefficient, see formular above.

    Returns:
        kernel: kernel function, to be applied to two arrays of vectors (yielding a 2-array corresponding all combinations of the input vectors).
    '''
    def __init__(self, gamma=1):
        self.gamma = gamma
    def __call__(self, X, Y=None):
        K = polynomial_kernel(1, 0, self.gamma)(X,Y)
        return K
    def diag(self, X):
        norm_squared = np.linalg.norm(X, axis=1)**2
        return self.gamma * norm_squared
    def is_stationary(self):
        return False

# prepare data:

def read_compounds(file_name, names_first=True, skip_parameters=0):
    '''Reads compound fingerprint data from a csv file into arrays.
    Arguments:
              file_name: filename (with pathg if needed) of the data file to be imported.
            names_first: Boolean wether the first column of the csv should be skipped. Some files have the names of the compounds there. Default is True.
        skip_parameters: int, number of columns to be skipped except the name column and the following label column. Some files may have information about the labels in multiple columns. Default is 0.

    Returns:
                 matrix: fingerprint data of the compounds as an array of vectors, of shape (rows, d).
                 labels: list of labels of the compounds, binary in [-1,+1].
                 names: list of names of the compounds of length rows. If names_first =  False, standartized names are chosen.
    '''
    scan = np.genfromtxt(file_name, delimiter=";")
    rows = len(scan)-1
    cols = len(scan[0])
    matrix = np.genfromtxt(file_name, delimiter=";", skip_header=1, usecols=range(1 + int(names_first) + skip_parameters, cols))
    if names_first:
        names = np.genfromtxt(file_name, delimiter=";", skip_header=1, usecols=0, dtype="S")
    else: 
        names = np.array(['Compound ' + str(i) for i in range(rows)])
    classes = np.genfromtxt(file_name, delimiter=";", skip_header=1, usecols=int(names_first), dtype="S") 
    labels = [+1 if (s==b'"Approved"' or s==b'1') else -1 if (s==b'"Withdrawn"' or s==b'0' or s==b'-1') else 0 for s in classes]
    return(matrix, labels, names)

def read_parquet(file_name):
    '''Reads compound fingerprint data from a parquet file into arrays.
    Arguments:
              file_name: filename (with pathg if needed) of the data file to be imported.

    Returns:
                 matrix: fingerprint data of the compounds as an array of vectors, of shape (rows, d).
                 labels: array of labels of the compounds, binary in [-1,+1].
                 names: array of names of the compounds.
    '''
    df = pd.read_parquet(file_name)
    smiles=df['Drug']
    fps=[]
    for s in smiles:
        rdkit.RDLogger.DisableLog('rdApp.*')  # suppress depreciation warnings
        mol = rdkit.Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        fps.append(fp)
    matrix = np.array(fps)
    matrix_labels = 2*df['Y'].to_numpy() - 1
    names = df['Drug_ID'].to_numpy()
    return matrix, matrix_labels, names

def prepare_data_sets(vectors, labels, train_percentage=80, positive_negative_ratio=None, max_train_size=None, min_train_size=None, normalize_data=False, seed=None, print_info=True):
    """Calculates a random instance of training and test data sets of certain sizes and properties. If needed, oversamples the minority class.
    Arguments:
                        vectors: array of data vectors, of shape (N,d) with d>1.
                         labels: list or array of binary (1,-1) labels of shape (N).         
               train_percentage: float: percentage (in (0,100])) of how much of the data should be used as training data. The remaining fraction will be used as test data. Default of 80 corresponds to a 80-20 split.
        positive_negative_ratio: ratio (bias) between positively labelled and negatively labelled vectors in both the training and test sets. A neutral bias corresponds to 1. None keeps the bias original bias.
                 max_train_size: upper bound for the size of training set. This bound is only contested by data sets where no oversampling is necessary. None removes the bound.
                 min_train_size: lower bound for the size of training set. To meet this bound with the specified positive_negative_ratio, oversampling of the minority class might be employed. None removes the bound and thus prohibits oversampling.
                 normalize_data: Boolean wether data should be normalized with the sklearn StandardScaler (centroid = 0 and standard deviation = 1). The test data will be normalized according to the centroid and standard deviation of the training data.
                           seed: None or int. Seed for randomness. The same seed should repeatedly give the same output.
                     print_info: Boolean if info should be printed.
    
    Returns:
|                       vectors: array of training data vectors.
                         labels: array of training labels.
                   test_vectors: array of test data vectors.
                    test_labels: array of test labels.
    """
    assert 0 < train_percentage, 'Training set can not be empty. Increase train_percentage. If you want only a test set, specify train_percentage=100 and switch the returns.'
    if isinstance(max_train_size, int) and isinstance(min_train_size, int):
        assert max_train_size >= min_train_size, 'specified max training set size smaller than min training set size.'
    # force binary labels:
    for i, label in sorted(enumerate(labels), reverse=True):
        if label != 1 and label != -1:
            vectors = np.delete(vectors, (i), axis=0)
            labels = np.delete(labels, (i))
    m = len(vectors)
    if print_info:
        print('Data size: {} ({} positive, {} negative)'.format(m, sum(1 for label in labels if label > 0), sum(1 for label in labels if label < 0)))
    # remove degenerate vectors with inconsistent labels:
    uniques, counts = np.unique(vectors, axis=0, return_counts=True)
    degeneracies = uniques[counts>1]
    for vector in degeneracies:
        deg_indices = list(np.where(np.all(vectors == vector, axis=1))[0])
        deg_labels = np.array(labels)[deg_indices]
        if not np.all(np.full(np.shape(deg_labels), deg_labels[0]) == deg_labels):
            for i in deg_indices[::-1]:
                vectors = np.delete(vectors, (i), axis=0)
                labels = np.delete(labels, (i))
    if print_info:
        print('{} vectors removed due to inconsistently labelled degeneracies'.format(m - len(vectors)))
    # shape dataset to conform with chosen bias positive_negative_ratio:
    num_positive = np.count_nonzero(labels > 0)
    num_negative = len(vectors) - num_positive
    current_pos_neg_ratio = num_positive / num_negative
    if positive_negative_ratio:
        if positive_negative_ratio > current_pos_neg_ratio:
            assert current_pos_neg_ratio <= 1 and positive_negative_ratio <= 1, 'You are trying to increase the bias of the dataset. Please decrease it by choosing a value of positive_negative_ratio between {} and 1 or keep the current bias by setting it to None.'.format(current_pos_neg_ratio)
            if isinstance(min_train_size, (int, float)) and num_positive * (1 + 1/positive_negative_ratio) < min_train_size * 100/train_percentage: # add positives by oversampling
                # but first reduce some negative vectors for not needing to oversample so much:
                pos_vectors = np.array([vector for vector, label in zip(vectors, labels) if label > 0])
                neg_vectors = np.array([vector for vector, label in zip(vectors, labels) if label < 0])
                reduced_neg_vectors, _, _, _ = skl.model_selection.train_test_split(neg_vectors, -np.ones(len(neg_vectors)), test_size=None, train_size=int(min_train_size * 100/train_percentage / (1+1/positive_negative_ratio)) + 1, random_state=seed)
                n = len(reduced_neg_vectors) + len(pos_vectors)
                vectors, labels = imb.over_sampling.SMOTE(sampling_strategy=positive_negative_ratio, random_state=seed).fit_resample(np.append(pos_vectors, reduced_neg_vectors, axis=0), np.append(np.ones(len(pos_vectors)), -np.ones(len(reduced_neg_vectors))))
                if print_info:
                    print('Added {} synthetic positives by oversampling'.format(len(vectors) - n))
            else: # delete negatives by undersampling
                vectors, labels = imb.under_sampling.ClusterCentroids(sampling_strategy=positive_negative_ratio, random_state=seed).fit_resample(vectors, labels)
        elif positive_negative_ratio < current_pos_neg_ratio:
            assert current_pos_neg_ratio >= 1 and positive_negative_ratio >= 1, 'You are trying to increase the bias of the dataset. Please decrease it by choosing a value of positive_negative_ratio between 1 and {} or keep the current bias by setting it to None.'.format(current_pos_neg_ratio)
            if isinstance(min_train_size, (int, float)) and num_negative * (1 + positive_negative_ratio) < min_train_size * 100/train_percentage: # add negatives by oversampling
                # but first reduce some negative vectors for not needing to oversample so much:
                pos_vectors = np.array([vector for vector, label in zip(vectors, labels) if label > 0])
                neg_vectors = np.array([vector for vector, label in zip(vectors, labels) if label < 0])
                reduced_pos_vectors, _, _, _ = skl.model_selection.train_test_split(pos_vectors, np.ones(len(pos_vectors)), test_size=None, train_size=int(min_train_size * 100/train_percentage / (1+positive_negative_ratio)) + 1, random_state=seed)
                n = len(reduced_pos_vectors) + len(neg_vectors)
                vectors, labels = imb.over_sampling.SMOTE(sampling_strategy=1/positive_negative_ratio, random_state=seed).fit_resample(np.append(reduced_pos_vectors, neg_vectors, axis=0), np.append(np.ones(len(reduced_pos_vectors)), -np.ones(len(neg_vectors))))
                if print_info:
                    print('Added {} synthetic negatives by oversampling'.format(len(vectors) - n))
            else: # delete positives by undersampling
                vectors, labels = imb.under_sampling.ClusterCentroids(sampling_strategy=1/positive_negative_ratio, random_state=seed).fit_resample(vectors, labels)
    # split in training and test sets:
    if train_percentage >= 100:
        vectors, labels, test_vectors, test_labels = *skl.utils.shuffle(vectors, labels), np.atleast_2d([]), np.atleast_1d([])
    else:
        max_relative_train_size = train_percentage/100
        relative_train_size = min(max_relative_train_size, max_train_size / len(vectors)) if isinstance(max_train_size, int) else max_relative_train_size
        relative_test_size = min(relative_train_size / (train_percentage / (100 - train_percentage)), 1 - relative_train_size)
        assert len(vectors) >= min_train_size * 100/train_percentage if isinstance(min_train_size, (int, float)) else True, 'Not enough data to meet chosen minumum training set size.'
        vectors, test_vectors, labels, test_labels = skl.model_selection.train_test_split(vectors, labels, test_size=relative_test_size, train_size=relative_train_size, random_state=seed)
    # sort test_vectors after test_labels:
    if test_vectors.size > 0:
        test_vectors, test_labels = zip(*sorted(zip(test_vectors, test_labels), key=lambda x: x[1], reverse=True))
    if normalize_data:
        scaler = skl.preprocessing.StandardScaler()
        vectors = scaler.fit_transform(vectors)
        if test_vectors.size > 0:
            print(test_vectors)
            test_vectors = scaler.transform(test_vectors)
    if print_info:
        print('Training data size: {} ({} positive, {} negative)'.format(len(vectors), sum(1 for label in labels if label > 0), sum(1 for label in labels if label < 0)))
    return(np.array(vectors), np.array(labels), np.array(test_vectors), np.array(test_labels))

def slice_training_data(training_vectors, training_labels, slice_size=None, force_unbiased=False, print_info:bool=False, seed=None):
    """Randomly slices a training data set with bias into smaller equal size pieces without repeating vectors of the bias class. There is the option to choose the minority class vectors such that their centroid within a slice is close or far away from the centroid of the majority class data. 
    Arguments:
                      training_vectors: array of training vectors to slice, of shape (N,d).
                       training_labels: list or array of binary [-1,+1] labels of training vectors, of shape (N).
                            slice_size: int > 1, desired size of slices. Default is None, which will choose the slices as large as possible.
                        force_unbiased: Boolean wether each slices should be balanced with respect to positive and negative data. If true, the vectors of the minority class will appear in multiple slices. Default is False.
                            print_info: Boolean wether info should be printed.
                                  seed: None or int. Seed for randomness. The same seed should repeatedly give the same output.
    
    Returns:
                                slices: array of data slices of shape (num_slices,N,d).
                          slice_labels: array of labels of data vectors in data slices, of shape (num_slices,N).
                                counts: array of counts of appearences of negative vectors across all slices. If random_type = 'equal', these counts will differ by at most 1.
|                       
    """
    ## this function assumes that we have a bias in the data. Otherwise, it may do weird things
    np.random.seed(seed)
    N, d = training_vectors.shape
    if force_unbiased:
        # assume that we have positive bias:
        flip_bias = False
        if np.count_nonzero(training_labels > 0) < np.count_nonzero(training_labels < 0):
            flip_bias = True
            training_labels = - np.array(training_labels)
        pos_vectors = np.array([vector for vector, label in zip(training_vectors, training_labels) if label > 0])
        neg_vectors = np.array([vector for vector, label in zip(training_vectors, training_labels) if label < 0])
        assert len(neg_vectors) > 0, 'Please include data with negative labels.'
        neg_overall_centroid = np.mean(neg_vectors, axis=0)
        slice_size_fixed = True if slice_size else False
        if slice_size is None:
            slice_size = 2 * len(neg_vectors) + 1
        while True:
            # distribute positive vectors:
            num_slices = round(2*len(pos_vectors)/slice_size)
            num_positive_per_slice = len(pos_vectors) // num_slices * np.ones(num_slices, dtype=int) + np.append(np.ones(len(pos_vectors) % num_slices, dtype=int), np.zeros(num_slices - len(pos_vectors) % num_slices, dtype=int))
            pos_vectors = skl.utils.shuffle(pos_vectors, random_state=seed)
            slices = [np.append(pos_vectors[int(np.sum(num_positive_per_slice[:i])):int(np.sum(num_positive_per_slice[:i]) + num_positive)], np.zeros((int(slice_size - num_positive),d)), axis=0) for i, num_positive in enumerate(num_positive_per_slice)]
            if len(neg_vectors) >=  np.max(slice_size - num_positive_per_slice) or slice_size_fixed:
                break
            slice_size -= 1
        assert slice_size > 1, 'Please choose more than just one vector per slice'
        assert len(neg_vectors) >=  np.max(slice_size - num_positive_per_slice), 'Choose smaller slices, such that we have enough negative data to fill them.'
        # complement with random instances of neg_vectors:
        for i, num_positive in enumerate(num_positive_per_slice):
            neg_slice = neg_vectors[np.random.choice(len(neg_vectors), int(slice_size - num_positive), replace=False)]
            slices[i][num_positive:] = neg_slice
        # extract labels
        slice_labels = np.array([np.append(np.ones(num_positive), -np.ones(slice_size - num_positive)) for num_positive in num_positive_per_slice])
        for i in range(len(slices)):
            slices[i], slice_labels[i] = skl.utils.shuffle(slices[i], slice_labels[i])
        if flip_bias == True:
            slice_labels *= -1
        # info about minority slice distribution:
        counts = [np.count_nonzero([np.any(np.all(slice == vector, axis=1)) for j, slice in enumerate(slices)]) for i, vector in enumerate(neg_vectors)]
        if print_info:
            print('Variance in negative vector appearences: ', np.var(counts))
    else:
        if slice_size is None:
            slice_size = N
        training_vectors, training_labels = skl.utils.shuffle(training_vectors, training_labels, random_state=seed)
        num_slices = np.ceil(N / slice_size)
        slices = np.array_split(training_vectors, num_slices)
        slice_labels = np.array_split(training_labels, num_slices)
        counts = None
    return(slices, slice_labels, counts)

# plots:

def plot_compounds(decision_values, labels, highlight=[], highlight_text='', marker_size=20, aspect_ratio=None, save=False):
    """Illustrates the classification of labelled data. 
    Arguments:
        decision_values: array of classification values of data set of shape (N).
                 labels: list or array of labels of data vectors in matrix, of shape (N).
              highlight: list, subset of range(N), containing vectors which are to be highlighted in the plot.
         highlight_text: Legend text giving information about highlighted vectors.
            marker_size: float, sizes of markers.
           aspect_ratio: float specifying the aspect ration of the plot. If the standard of None is chosen, matplotlib chooses a ratio.
                   save: Boolean wether the plot is to be saved or string specifying the file name (assuming it is to be saved).

    Returns:
        -
    """
    M = len(decision_values)
    x = range(M)
    y = decision_values
    if highlight:
        normal = np.setdiff1d(x, highlight)
        y_highlighted = np.array(y)[highlight]
        y_normal = np.array(y)[normal]
        plt.scatter(normal,y_normal, s=marker_size, c=['blue' if labels[i]==+1 else 'red' if labels[i]==-1 else 'grey' for i in normal], marker='o')
        plt.scatter(highlight,y_highlighted, s=marker_size, c=['blue' if labels[i]==+1 else 'red' if labels[i]==-1 else 'grey' for i in highlight], marker='x', label=highlight_text)
        plt.legend()
    else:
        plt.scatter(x,y, s=marker_size, c=['blue' if labels[i]==+1 else 'red' if labels[i]==-1 else 'grey' for i in range(M)])
    ax = plt.gca()
    ax.axhline(y=0, c='gray', alpha=0.5, linestyle='--') # origin axis line
    plt.ylabel('f')
    if isinstance(aspect_ratio, (int, float)):
        ax.set_aspect(abs((ax.get_xlim()[0] - ax.get_xlim()[1])/(ax.get_ylim()[0]-ax.get_ylim()[1]))*aspect_ratio)
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png' ,dpi=300)
    plt.show()

def boxplot_compounds(decision_values, labels, save=False):
    """Illustrates the classification of the label classes [-1,0,+1] of a labelled data set. This plot basically turns plot_compounds into a boxplot
    Arguments:
        decision_values: array of classification values of data set of shape (N).
                 labels: list or array of labels of data vectors in matrix, of shape (N).
                   save: Boolean wether the plot is to be saved or string specifying the file name (assuming it is to be saved).

    Returns:
        -
    """
    plt.boxplot(x=[
        [values for i, values in enumerate(decision_values) if labels[i] == +1], # approved
        [values for i, values in enumerate(decision_values) if labels[i] == -1], # withdrawn
        [values for i, values in enumerate(decision_values) if labels[i] == 0] if np.count_nonzero(labels == 0) > 0 else [], # preclinical
    ])
    ax = plt.gca()
    ax.axhline(y=0, c='gray', alpha=0.5, linestyle='--') # origin axis line
    ax.set_xticks(range(1,4), [
        'approved',
        'withdrawn',
        'preclinical',
    ], fontsize=7)
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png' ,dpi=300)
    plt.show()

def boxplot_compounds2(decision_values, labels, function_labels=['qSVM', 'cSVM'], save=False):
    """Illustrates the two classifications of the label classes [-1,0,+1] of a labelled data set. This basically doubles the boxplot_compounds function.
    Arguments:
        decision_values: array of classification values of data set of shape (N).
                 labels: list or array of labels of data vectors in matrix, of shape (N).
        function_labels: list of length 2 of labels of the decision functions. Default is ['qSVM', 'cSVM'], comparing the classical and quantum approaches.
                   save: Boolean wether the plot is to be saved or string specifying the file name (assuming it is to be saved).

    Returns:
        -
    """
    fig, ax = plt.subplots(1,2)
    for j in range(2):
        ax[j].boxplot(x=[
            [decision_values[j,i] for i, label in enumerate(labels) if label == +1], # approved
            [decision_values[j,i] for i, label in enumerate(labels) if label == -1], # withdrawn
            [decision_values[j,i] for i, label in enumerate(labels) if label == 0], # preclinical
        ])
        ax[j].axhline(y=0, c='gray', alpha=0.5, linestyle='--') # origin axis line
        ax[j].set_box_aspect(0.7)
        ax[j].set_xticks(range(1,4), [
            'approved',
            'withdrawn',
            'preclinical',
        ], fontsize=7)
        ax[j].title.set_text(function_labels[j])
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png' ,dpi=300)
    plt.show()

def plot_curve(x, y, xlabel, ylabel, save=False):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png' ,dpi=300)
    plt.show()

def plot_training_data_with_decision_boundary(classifyer, X=None, y=None, plot_margins=True, title='Decision_boundary', support_vectors=False, save=False):
    """Plots the decision boundary of a classifyer with data.
    Arguments:
        classifyer: estimator object of classifying function. If it is already fitted and has attributes 'X_' and 'y_', those will be used as the second and third argument.
        X: array of shape (N,d) of data vector the classifyier was trained on or can be trained on. Should be compatible with the classifyer. Default is None, which is only possible if the classifyer is a function or already has been trained.
        y: array or list of length N of labels of X. Default is None, which is only possible if the classifyer is a function or already has been trained. If X is not None, y will then take the predicted labels of X.
        plot_margins: Boolean wether margins of the decision boundary should be plotted. This seems to not always work well. Default is True.
        title: string as title for the plot. Default is 'Decision_boundary'.
        support_vectors: Boolean wether support vectors should be plotted. This does not always seem to work. Default is False.
        save: Boolean wether the plot is to be saved or string specifying the file name (assuming it is to be saved).

    Returns:
        -
    """
    assert hasattr(classifyer, 'fit') or hasattr(classifyer, 'decision_function') or callable(classifyer), 'Please specify an estimator object or a classifying function as first argument.'
    if callable(classifyer): # build a custom estimator out of classifying function
        clf = manual_classifier(decision_function_=classifyer, X_=X, y_=y)
    elif hasattr(classifyer, 'fit') or hasattr(classifyer, 'decision_function'):
        # Train the SVC, if needed
        if getattr(classifyer, 'is_fitted_', None):
            if X is None and y is None and hasattr(classifyer, 'X_') and hasattr(classifyer, 'y_'):
                X, y = classifyer.X_, classifyer.y_
            clf = classifyer
        else:
            clf = classifyer.fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots()

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        cmap=ListedColormap(["red", "blue"])
    )
    
    if plot_margins:
        DecisionBoundaryDisplay.from_estimator(
            **common_params,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
        )
    else:
        DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors=["k"],
        linestyles=["-"],
        )
     
    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(title)
    if save:
        if not isinstance(save, (str)): # standard name
            save = 'plot'
        plt.savefig(save + '.png' ,dpi=300)
    plt.show()

# evaluation metrics:

def kappa(decision_values, test_labels):
    """Computes the Cohen's Kappa score of a classification of labelled data. The Kappa score is calculated by (p_o - p_e) / (1 - p_e), where p_o is the empirical probability of agreement (the observed agreement ratio) and p_e is the expected agreement when labels are assigned randomly.
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
        
    Returns:
         kappa_score: Cohen's Kappa score.
    """
    predicted_labels = np.sign(decision_values)
    kappa_score = skl.metrics.cohen_kappa_score(test_labels, predicted_labels, labels=np.array([-1, 1]))
    return(kappa_score)
def accuracy(decision_values, test_labels):
    """Computes the Accuracy score ((TP+TN) / (TP+TN+FP+FN)) of a classification of labelled data.
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
        
    Returns:
        accuracy_score: accuracy score.
    """
    predicted_labels = np.sign(decision_values)
    accuracy_score = skl.metrics.accuracy_score(test_labels, predicted_labels)
    return(accuracy_score)
def AUROC(decision_values, test_labels, plot=False):
    """Computes the area under Receiver Operating Characteristic (ROC) curve of classification of labelled data. The ROC curve is obtained by plotting the True Positive Rate (TPR) vs. the False Positive Rate (FPR) while sweeping bias b. See the function ROC_curve for more info.
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
                   plot: Boolean wether the ROC curve should be plotted.
        
    Returns:
        auroc: area under ROC curve.
    """
    auroc = skl.metrics.roc_auc_score(test_labels, decision_values)
    if plot:
        false_positive_rate, true_positive_rate = ROC_curve(decision_values, test_labels)
        plot_curve(false_positive_rate, true_positive_rate, 'False Positive Rate', 'True Positive Rate')
    return(auroc)
def AUPRC(decision_values, test_labels, plot=False):
    """Computes the area under Precision Recall (PR) curve of a classification of labelled data. The PR curve is obtained by plotting the Precision vs. the Recall (True Positive Rate) while sweeping bias b. See the function PR_curve for more info.
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
                   plot: Boolean wether the PR curve should be plotted.
        
    Returns:
               auprc: area under PR curve.
    """
    recall, precision= PR_curve(decision_values, test_labels)
    auprc = skl.metrics.auc(recall, precision)
    # auprc = np.trapz(precision, recall) # this seems to yield the same but is faster
    if plot:
        plot_curve(recall, precision, 'Recall', 'Precision')
    return(auprc)
def ROC_curve(decision_values, test_labels):
    """Computes the Receiver Operating Characteristic (ROC) curve of a classification of labelled data. It is obtained by plotting the True Positive Rate (TPR) vs. the False Positive Rate (FPR) while sweeping bias b. TPR = TP / (TP+FN), FPR = FP / (FP+TN).
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
        
    Returns:
        false_positive_rate: FPR (x-axis).
         true_positive_rate: TPR (y-axis).
        
    """
    false_positive_rate, true_positive_rate, _ = skl.metrics.roc_curve(test_labels, decision_values)
    return(false_positive_rate, true_positive_rate)
def PR_curve(decision_values, test_labels):
    """Computes the Precision Recall (PR) curve of a classification of labelled data. It is obtained by plotting the Recall (True Positive Rate (TPR)) vs. the False Positive Rate (FPR) while sweeping bias b. Recall = TPR = TP / (TP+FN), Precision = TP / (TP+FP).
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
        
    Returns:
              recall: TPR (x-axis).
           precision: Presicion (y-axis).
        
    """
    precision, recall, _ = skl.metrics.precision_recall_curve(test_labels, decision_values)
    return(recall[::-1], precision[::-1])

def print_measures(decision_values, test_labels, title=''):
    """Prints the Kappa score, Accuracy score, AUROC and AUPRC of a classification of labelled data.
    Arguments:
        decision_values: array of classification values of data set of shape (N). A positive value is interpreted as +1, while a negative value yields label -1.
            test_labels: array or list of labels of vectors, of shape (N).
                  title: title to be printed vefore the measures. Default is ''.
        
    Returns:
        -
    """
    print(title)
    print('Kappa score = ', kappa(decision_values, test_labels))
    print('Accuracy score = ', accuracy(decision_values, test_labels))
    print('AUROC = ', AUROC(decision_values, test_labels))
    print('AUPRC = ', AUPRC(decision_values, test_labels))

# further functions:
    
def distribute_items_equally(items, bin_sizes):
    # Initialize bin contents and item usage counters
    bins = [[] for _ in range(len(bin_sizes))]
    item_usage = {item: 0 for item in items}
    for bin_index, bin_size in enumerate(bin_sizes):
        while len(bins[bin_index]) < bin_size:
            # Filter items that are not already in the current bin
            available_items = [item for item in items if item not in bins[bin_index]]
            if not available_items:
                raise ValueError("Not enough unique items to fill the bin")
            # Select item with the minimum usage count from available items
            min_usage = min(item_usage[item] for item in available_items)
            candidates = [item for item in available_items if item_usage[item] == min_usage]
            # Randomly select from the least used items
            selected_item = np.random.choice(candidates)
            # Assign the selected item to the bin
            bins[bin_index].append(selected_item)
            item_usage[selected_item] += 1
    return(bins)

# define estomator classes to use the custom algorithms in sklearn methods:

class cSVM_estimator(skl.base.BaseEstimator):
    """Estimator class to classically train a SVM by solving the corresponding quadratic optimization problem.
    Arguments:
        upper_bound: upper bound of the (positive) training coefficients. Should not be too little but does not have much impact on the results.
             kernel: information specifying the kernel used to transform the data into one dimension. The options are
                             None: radial kernel with standard gamma coefficient value of 1 / (d * vectors.var()). This option is the default.
                            float: gamma coefficient to be used with radial kernel.
                            tuple: of string in ['rbf', 'radial', 'gauss', 'poly', 'linear'] (the first 3 being the same) and corresponding coefficients, these can be viewed as the arguments of the functions 'gauss_kernel', 'polynomial_kernel' and 'linear_kernel'. 
                callable function: a function mapping arrays of vectors (of shapes (N,d)) onto an array of shape (N,N). This (n,m)-th element corresponds to the kernel acting on the n-th and m-th vectors. Note that this option does not work when the solver 'sklearn' is chosen.
             solver: information specifying the method of training (optimization). The options are
                string: 'CPLEX' or 'Bonmin' for Pyomo solvers (works only if installed). 'sklearn' or 'svc' for standard SVC algorithm by scikit learn. 'hybrid' for D-Wave hybrid sampler (This does not work for now, unless D-Wave changes something). 
                tuple: of string as above and computation time limit, except for when using scikit learn.
            samples: number of samples of solutions (classifiers) to be returned. For classical methods the number will always be one.
        adjust_bias: Boolean wether to adjust bias in a brute force search after optimization, or string. String means True but with a specified metric ('kappa' for kappa, else accuracy is used). Does not work with the scikit learn solver. Generally it is not recommended, as it is unlikely to improve on the results.
         print_info: Boolean wether info should be printed.

    Returns:
        estimator object with methods fit, predict and decision_function.
    """
    def __init__(self, *, upper_bound=100, kernel=None, solver='svc', adjust_bias=False, print_info:bool=False):
        self.upper_bound = upper_bound
        self.kernel = kernel
        self.solver = solver
        self.adjust_bias = adjust_bias
        self.print_info = print_info
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = skl.utils.validation.check_X_y(X, y)
        skl.utils.multiclass.check_classification_targets(y)
        # Validate that y contains only -1 and 1
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [-1, 1]):
            raise ValueError(f"Labels should be -1 or 1. Found labels: {unique_labels}")
        # Store the classes seen during fit
        self._estimator_type = "classifier"
        self.classes_ = skl.utils.multiclass.unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        # Training:
        objective, alpha, bias, f = cSVM(X, y, upper_bound=self.upper_bound, kernel=self.kernel, solver=self.solver, adjust_bias=self.adjust_bias, print_info=self.print_info)
        self.objective_ = objective
        self.alpha_ = alpha
        self.bias_ = bias
        self.decision_function_ = f
        self.is_fitted_ = True
        return self
    def predict(self, X):
        skl.utils.validation.check_is_fitted(self)
        X = skl.utils.validation.check_array(X)
        if self.alpha_ is None:
            decision_value = estimator_add_bias(X, self.decision_function_, self.bias_)
        else:
            decision_value = estimator_decision_function(X, self.X_, self.y_, self.alpha_, self.bias_, self.kernel, self.solver)
        return np.sign(decision_value)
    def decision_function(self, X):
        skl.utils.validation.check_is_fitted(self)
        if self.alpha_ is None:
            return estimator_add_bias(X, self.decision_function_, self.bias_)
        else:
            return estimator_decision_function(X, self.X_, self.y_, self.alpha_, self.bias_, self.kernel, self.solver)
    
class qSVM_estimator(skl.base.BaseEstimator):
    """Estimator class to quantumly train a SVM by converting the corresponding quadratic optimization problem into QUBO format and solving it with annealing methods.
    Arguments:
                   base: integer base B of binary encoding used. Determines the spacing between possible values of the variables. Default: 2 (regular integer spacing).
           num_encoding: integer number K of binaries used to incode a real number. Determines the amount 2^K of possible values of the variables. The upper bound of the variables will correspond to \sum_k^K B^k. Default: 3
                penalty: penalty weights for the penalty terms encoding the constraint of the classical SVM formulation. Setting penalty = False and choosing the hybrid solver, the constraint will be handelled by the solver. Standard of True corresponds to zero penalty.
                 kernel: information specifying the kernel used to transform the data into one dimension. The options are
                                 None: radial kernel with standard gamma coefficient value of 1 / (d * vectors.var()). This option is the default.
                                float: gamma coefficient to be used with radial kernel.
                                tuple: of string in ['rbf', 'radial', 'gauss', 'poly', 'linear'] (the first 3 being the same) and corresponding coefficients, these can be viewed as the arguments of the functions 'gauss_kernel', 'polynomial_kernel' and 'linear_kernel'. 
                    callable function: a function mapping arrays of vectors (of shapes (N,d)) onto an array of shape (N,N). This (n,m)-th element corresponds to the kernel acting on the n-th and m-th vectors.
                 solver: information specifying the method of training (optimization). The options are
                    string: 
                                    'SA': simulated annealing (standard).
                                'hybrid': D-Wave Leap hybrid solver, either as QUBO (penalty != False), or as a CQM (penalty = False).
                             'annealing': D-Wave sampler for quantum annealing. Will look for an embedding, which may take up to 1 minute and perhaps does not terminate.
                                'clique': Clique sampler for quantum annealing. Will immediately find an embedding if the QUBO dimension is less or equal than 177.
                                  'lazy': LazyFixedEmbeddingComposite sampler for quantum annealing. Makes it possible to specify embeddings and/or save them. Use this sampler if you want to reuse the embedding, for example if you cast the same training with a different penalty.
                        'save_embedding': LazyFixedEmbeddingComposite sampler for quantum annealing. Will return the embedding after the training.
                     float: Cutoff between 0 and 1. The relative desired increase in zeros in the QUBO. This it will ignore some terms such that the QUBO becomes more sparse. 1 would mean that all non-linear terms are 0. If an embedding is specified or to be saved, then the LazyFixedEmbeddingComposite sampler is used, otherwise the CutOffComposite sampler.
                     tuple: of string or float as above and an integer. For hybrid solver the integer describes the computation time limit (> 5). For all other samplers it is the number of shots (standard 100).
                samples: number of samples of solutions (classifiers) to be returned. Default is 20.
        choose_function: 'lowest' or 'mean', signifying the way to choose the final decision function from the set of samples. Default is 'mean'.
            adjust_bias: Boolean wether to adjust bias in a brute force search after optimization or string. String means True but with a specified metric ('kappa' for kappa, else accuracy is used). It seems to yield an improvement on sampling methods quite often. Default is False.
              embedding: object of type dwave.embedding.transforms.EmbeddedStructure, which would force the LazyFixedEmbeddingComposite sampler with the specified embedding or 'save', which will return the previously unspecified embedding used in the training as an dwave.embedding.transforms.EmbeddedStructure object. Default is None.
             print_info: Boolean wether info should be printed.

    Returns:
        estimator object with methods fit, predict and decision_function.
    """
    def __init__(self, *, base=2, num_encoding=3, penalty=True, kernel=None, solver='SA', samples=20, choose_function='mean', adjust_bias=False, embedding=None, print_info:bool=False):
        self.base = base
        self.num_encoding = num_encoding
        self.penalty = penalty
        self.kernel = kernel
        self.solver = solver
        self.samples = samples
        self.choose_function = choose_function
        self.adjust_bias = adjust_bias
        self.embedding = embedding
        self.print_info = print_info
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = skl.utils.validation.check_X_y(X, y)
        # Store the classes seen during fit
        self._estimator_type = "classifier"
        self.classes_ = skl.utils.multiclass.unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        # Training:
        objectives, alphas, biases, fs, embedding = qSVM(X, y, base=self.base, num_encoding=self.num_encoding, penalty=self.penalty, kernel=self.kernel, solver=self.solver, samples=self.samples, adjust_bias=self.adjust_bias, embedding=self.embedding, print_info=self.print_info)
        self.objectives_ = objectives
        self.mean_objective_ = np.mean(objectives)
        self.alphas_ = alphas
        self.decision_functions_ = fs
        self.biases_ = biases
        self.embedding = embedding
        if self.choose_function == 'lowest':
            self.bias_ = self.biases_[0]
            self.alpha_ = self.alphas_[0]
        else:
            self.bias_ = np.mean(self.biases_)
            self.alpha_ = np.mean(self.alphas_, axis=0)
        self.is_fitted_ = True
        return self
    def predict(self, X):
        skl.utils.validation.check_is_fitted(self)
        X = skl.utils.validation.check_array(X)
        decision_value = estimator_decision_function(X, self.X_, self.y_, self.alpha_, self.bias_, self.kernel, self.solver)
        return np.sign(decision_value)
    def decision_function(self, X):
        skl.utils.validation.check_is_fitted(self)
        return estimator_decision_function(X, self.X_, self.y_, self.alpha_, self.bias_, self.kernel, self.solver)
    
class slice_estimator(skl.base.BaseEstimator):
    """Estimator class to divide training data in slices and train it using an estimator of choice
    Arguments:
         estimator: estimator object with methods fit, predict and decision_function.
        slice_size: int > 1, desired size of slices. Default is None, which means that the max possible slice slice will be chosen.
              seed: None or int. Seed for randomness. The same seed should repeatedly give the same output. Default is None
        print_info: Boolean wether info should be printed. Default is False.

    Returns:
        estimator object with methods fit and predict.
    """
    def __init__(self, *, estimator=skl.svm.SVC(), slice_size=None, force_unbiased=False, seed=None, print_info:bool=False):
        self.estimator = estimator
        self.slice_size = slice_size
        self.force_unbiased = force_unbiased
        self.seed = seed
        self.print_info = print_info
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = skl.utils.validation.check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = skl.utils.multiclass.unique_labels(y)
        self._estimator_type = "classifier"
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        # Slice training set:
        self.slice_vectors, self.slice_labels, counts = slice_training_data(X, y, self.slice_size, force_unbiased=self.force_unbiased, seed=self.seed, print_info=self.print_info)
        num_slices = len(self.slice_vectors)
        # loop over slices and train
        self.slice_estimators_ = [copy.deepcopy(self.estimator) for _ in range(num_slices)]
        for i, vectors, labels in zip(range(num_slices), self.slice_vectors, self.slice_labels):
            if self.print_info:
                print('Training slice: {}/{}'.format(i+1, num_slices), end="\r")
            self.slice_estimators_[i].fit(vectors, labels)
        self.is_fitted_ = True
        if self.print_info: # clear the previous print
            print(' ' * 25, end="\r")
        return self
    def predict(self, X):
        skl.utils.validation.check_is_fitted(self)
        X = skl.utils.validation.check_array(X)
        decision_value = np.mean([slice_estimator.decision_function(X) for slice_estimator in self.slice_estimators_], axis=0)
        return np.sign(decision_value)
    def decision_function(self, X):
        skl.utils.validation.check_is_fitted(self)
        return np.mean([slice_estimator.decision_function(X) for slice_estimator in self.slice_estimators_], axis=0)
    
def estimator_decision_function(X, vectors, labels, alpha, bias, kernel, solver):
    kernel, gamma, degree, coef0 = transform_kernel_argument(kernel, solver, vectors)
    assert kernel in ['rbf', 'linear', 'poly'] or callable(kernel), 'Invalid form of kernel specified'
    if kernel == 'rbf':
        kernel_terms = skl.metrics.pairwise.rbf_kernel(np.array(X), vectors, gamma=gamma)
    elif kernel == 'poly':
        kernel_terms = skl.metrics.pairwise.polynomial_kernel(np.array(X), vectors, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == 'linear':
        kernel_terms = skl.metrics.pairwise.polynomial_kernel(np.array(X), vectors, degree=1, gamma=gamma, coef0=0)
    elif callable(kernel):
        kernel_terms = kernel(np.array(X), vectors)
    f_value = np.einsum('i,i,ji->j', alpha, labels, kernel_terms) + bias
    return f_value

def estimator_add_bias(X, f, bias):
    f_value = f(X) + bias
    return(f_value)

def hyperparameter_optimization(estimator, param_grid:dict, vectors, labels, folds:int=5, filter=None, print_info=False):
    """Performs a hyperparameter optimization of an estimator object, in particular by cross validation with Stratified 5-Fold. Accuracy and Kappa value are applied as metric, and the optimization goes according to the Kappa value.
    Arguments:
                estimator: estimator object with methods fit, predict and decision_function.
               param_grid: dictionary with parameter names (str) as values and lists of values as values. Allowed parameters are:
                    - attributes of the estimator
                    - if estimator.kernel is callable and has attributes, the names of these attributes are allowed.
                    - if estimator is of class slice_estimator, the attributes of estimator.estimator are allowed
                  vectors: array of data vectors, of shape (N,d) with d>1.
                   labels: list or array of binary (1,-1) labels of shape (N).
                    folds: int, number of folds to use in the crossvalidation. Default is 5
                   filter: callable with boolean outputs and parameters in param_grid as inputs, filters the grid. Default is None, which means no filtering.
               print_info: Boolean wether info should be printed. Default is False.

    Returns:
        decision_function: classyfing function of the estimator fitted with the best parameters (with the best kappa value in the cross validation).
           df_nice_sorted: pandas data frame of hyperparameter values and associated metrics in the cross validation.
              grid_search: sklearn.model_selection.GridSearchCV object fitted with the best parameters. Useful attributes are cv_results_, best_estimator_, and best_params_.
    """
    scoring = {'accuracy': 'accuracy', 'kappa': skl.metrics.make_scorer(skl.metrics.cohen_kappa_score)}
    # filter grid:
    paramgrid = filter_grid(param_grid, filter) if callable(filter) else param_grid.copy() # copy to not change the original param_grid
    # check if kernel parameters have to be passed to the kernel parameters:
    def get_deep_attributes(estimator, attribute_name, condition=None):
        if condition is None:
            condition  = lambda *_: True
        inner_params = []
        if hasattr(estimator, attribute_name):
            if condition(getattr(estimator, attribute_name)):
                # Use inspect.signature to get the signature of the __init__ method
                sig = inspect.signature(getattr(estimator, attribute_name).__init__)
                # Exclude 'self' from the parameter list
                inner_params = [p for p in sig.parameters if p != 'self']
        return inner_params
    def extend_param_names(param_grid, param_names, prefix):
        grid_iterator = param_grid if isinstance(param_grid, list) else [param_grid]
        for param in param_names:
            for paramgrid in grid_iterator:
                if param in paramgrid:
                    paramgrid[prefix + '__' + param] = paramgrid.pop(param)
        return param_grid

    inner_kernel_params = get_deep_attributes(estimator, 'kernel', condition=callable)
    paramgrid = extend_param_names(paramgrid, inner_kernel_params, 'kernel')
    # check if estimator is the slice_estimator. If yes, we have to use a pipeline
    if isinstance(estimator, slice_estimator):
        inner_estimator_params = get_deep_attributes(estimator, 'estimator')
        paramgrid = extend_param_names(paramgrid, inner_estimator_params, 'estimator')

        inner_kernel_params = get_deep_attributes(estimator.estimator, 'kernel', condition=callable)
        paramgrid = extend_param_names(paramgrid, inner_kernel_params, 'estimator__kernel')

        pipeline = skl.pipeline.Pipeline([('sliced', estimator)])
        paramgrid = extend_param_names(paramgrid, set().union(*(d.keys() for d in paramgrid)) if isinstance(paramgrid, list) else paramgrid.keys(), 'sliced')
    else:
        pipeline = estimator
   # specify level of print detail
    assert isinstance(print_info, (bool, int)), 'Wrong format of print_info. Try again with bool or int.'
    if isinstance(print_info, bool):
        print_detail = 2 if print_info else 0
    elif isinstance(print_info, int):
        print_detail = print_info
     # run grid search with cross validation:
    grid_search = skl.model_selection.GridSearchCV(pipeline, paramgrid, scoring=scoring, cv=folds, refit='kappa', verbose=print_detail)
    grid_search.fit(vectors, labels)
    cv_results = grid_search.cv_results_
    # export dataframe:
    df_results = pd.DataFrame(cv_results)
    df_results = df_results.applymap(lambda x: x.filled(np.nan) if isinstance(x, np.ma.MaskedArray) else x)
    # Select relevant columns
    # Typically, we are interested in params and mean_test_score
    params = df_results.iloc[:, 1:].filter(like="param_")
    scores = df_results[['mean_test_kappa', 'std_test_kappa']]
    # Relabel:
    params.columns = params.columns.str.split('__').str[-1]
    scores.columns = scores.columns.str.replace('_test_', ' ')
    # Combine params and scores into a single DataFrame
    df_nice = pd.concat([params, scores], axis=1)
    df_nice_sorted = df_nice.sort_values(by='mean kappa', ascending=False)
    if print_info:
        display(df_nice_sorted)
    optimal_params = grid_search.best_params_
    decision_function = grid_search.best_estimator_.decision_function
    return(decision_function, df_nice_sorted, grid_search)

def filter_grid(param_grid, filter):
    """
    Filters parameter combinations based on a filter function and returns them 
    with values wrapped in lists, as expected by GridSearchCV.
    Parameters:
    - param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter values.
    - filter_func (function): A function that takes the parameters as arguments and returns True or False.
    Returns:
    - filtered_params (list): A list of dictionaries with valid parameter combinations,
                              each value wrapped in a list.
    """
    # Get all possible combinations from the parameter grid
    param_names = param_grid.keys()
    all_combinations = list(itertools.product(*param_grid.values()))
    # Filter combinations using the provided filter function
    filtered_combinations = [
        {param: [value] for param, value in zip(param_names, values)}
        for values in all_combinations
        if filter(**dict(zip(param_names, values)))
    ]
    return filtered_combinations

class manual_classifier(skl.base.BaseEstimator):
    """Estimator class that creates a classifier with a custom decision function, which already counts as fitted. This could be used to plot desicion_boundaries together with plot_training_data_with_decision_boundary.
    Arguments:
        decision_function: Custom decision_function. Default is 1, which yields the trivial positive classifier.
                       X_: (optional) array of vectors that was used to generate the decision_function, must be of the same shape that decision_function accepts. Default is None.
                       y_: (optional) array or list of labels in (-1,+1) that was used to generate the decision_function, must be of the same length as vectors. Default is None.
                   
    Returns:
        estimator object with methods fit, predict and decision_function.
    """
    def __init__(self, *, decision_function_=1, X_=None, y_=None):
        self.decision_function_ = decision_function_
        self.X_ = X_
        if X_ is not None and y_ is None:
            self.y_ = np.sign(decision_function_(X_))
        else:
            self.y_ = y_
        self._estimator_type = "classifier"
        self.classes_ = np.array([-1,  1])
        self.is_fitted_ = True
        self.kernel = None
    def decision_function(self, X):
        if isinstance(self.decision_function_, (int,float)):
            return self.decision_function_ * np.ones(len(X))
        else:
            return self.decision_function_(X)
    def fit(self, X, y):
        # Check that X and y have correct shape
        try:
            y_pred = np.sign(self.decision_function([X]))
            if self.X_ is None:
                self.X_ = X
                if self.y_ is None:
                    if y is None:
                        self.y_ = y_pred
                    else:
                        self.y_ = y
        except ValueError:
            raise ValueError('Use data of shape accroding to the custom decision function.')
    def predict(self, X):
        skl.utils.validation.check_is_fitted(self)
        X = skl.utils.validation.check_array(X)
        decision_value = self.decision_function(X)
        return np.sign(decision_value)
    
class mean_estimator(skl.base.BaseEstimator):
    """Estimator class that averages over estimators provided in its first argument.
    Arguments:
        estimators: list or array of estimators.

    Returns:
        estimator object with methods fit, predict and decision_function.
    """
    def __init__(self, *, estimators=[cSVM_estimator()]):
        assert np.all([hasattr(estimator, 'decision_function') for estimator in estimators]), 'All estimators have to have a decision function.'
        N = len(estimators)
        self.estimators = estimators
        def inherite_attributes_by_value(self, attr_name, attr_value, logic, exception):
            if logic([getattr(estimator, attr_name, None) == attr_value for estimator in self.estimators]):
                setattr(self, attr_name, attr_value)
            elif exception:
                setattr(self, attr_name, exception)
        def union_attributes(self, attr_name):
            union = []
            for arr in [getattr(estimator, attr_name, []) for estimator in self.estimators]:
                union = np.union1d(union, arr)
            attr_value = np.array(list(set(union)))
            setattr(self, attr_name, attr_value)
        def inherite_attributes_if_same(self, attr_name):
            inherite_attributes_by_value(self, attr_name, getattr(self.estimators[0], attr_name, None), np.all, False)
            
        union_attributes(self, 'classes_')
        for param in self.estimators[0].get_params():
            inherite_attributes_if_same(self, param)
        for attr in ['is_fitted_', '_estimator_type']:
            inherite_attributes_if_same(self, attr)
    def decision_function(self, X):
        mean = np.mean([estimator.decision_function(X) for estimator in self.estimators], axis=0)
        return mean
    def fit(self, X, y):
        assert np.all([hasattr(estimator, 'fit') for estimator in self.estimators]), 'All estimators have to have a decision function.'
        for estimator in self.estimators:
            estimator.fit(X,y)
    def predict(self, X):
        skl.utils.validation.check_is_fitted(self)
        X = skl.utils.validation.check_array(X)
        decision_value = self.decision_function(X)
        return np.sign(decision_value)
