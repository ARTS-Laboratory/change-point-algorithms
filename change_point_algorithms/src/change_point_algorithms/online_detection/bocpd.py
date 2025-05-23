import math
import warnings
from collections import deque

import numpy as np
from line_profiler import profile
from numba import njit, vectorize

try:
    from change_point_algorithms import change_point_algorithms
    from change_point_algorithms import run_bocpd, run_bocpd_inplace, DistParams, BetaCache, SparseProbs
except ModuleNotFoundError:
    warnings.warn('Rust module not included in environment.')
from change_point_algorithms.online_detection.model_helpers import (
    detection_to_intervals_for_generator_v1,
    detection_to_intervals_for_generator_v1_with_progress)


def bocpd_generator(data, mu, kappa, alpha, beta, lamb):
    """ Generator for Bayesian Online Change Point Detection Algorithm."""
    my_data = np.asarray(data)
    maxes = deque((0,), maxlen=2)
    run_length_arr = np.array([0], dtype=np.uint32)
    probabilities = np.array([1.0])
    alpha_arr, beta_arr, mu_arr, kappa_arr = np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa])
    for idx, event in enumerate(my_data):
        probabilities, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr = calculate_probabilities(
            event, alpha_arr, beta_arr, mu_arr, kappa_arr, run_length_arr, probabilities, lamb, trunc_threshold=1e-32)
        max_idx = find_max_cp(probabilities)
        maxes.append(run_length_arr[max_idx])
        if maxes[-1] < maxes[0]:
            # reset params
            probabilities = np.asarray([1.0])
            run_length_arr = np.asarray([0], dtype=np.uint32)
            # maxes = [0]
            alpha_arr, beta_arr, mu_arr, kappa_arr = (
                np.array([alpha]), np.array([beta]), np.array([mu]), np.array([kappa]))
        else:
            # update
            alpha_arr, beta_arr, mu_arr, kappa_arr = update_no_attack_arr(
                event, alpha_arr, beta_arr, mu_arr, kappa_arr, alpha, beta, mu, kappa)
        # Calculate probability of change point
        attack_probs = calculate_prior_arr(event, alpha_arr, beta_arr, mu_arr, kappa_arr)
        val_prob = np.dot(attack_probs, probabilities)
        is_attack = val_prob <= 0.05
        yield is_attack


def bocpd_rust_hybrid(data, mu, kappa, alpha, beta, lamb):
    """ Use Bayesian Online Change Point Detection to detect change points in data."""
    threshold = 1e-8
    my_data = np.asarray(data)
    run_length = 1  # Iterations since last changepoint
    maxes = deque((0,), maxlen=2)
    sparse_probs = SparseProbs()
    sparse_probs.new_entry(0, 1.0)
    parameters: DistParams = DistParams(alpha, beta, mu, kappa)
    beta_cache = BetaCache(0.5)
    for idx, event in enumerate(my_data):
        change_point_algorithms.calc_probabilities_cached(event, lamb, parameters, sparse_probs, beta_cache)
        # tda_project_rusty.calc_probabilities(event, lamb, parameters, sparse_probs)
        new_size = change_point_algorithms.truncate_vectors(threshold, parameters, sparse_probs)
        max_idx, _ = sparse_probs.max_prob()
        maxes.append(max_idx)
        if maxes[-1] < maxes[0]:
            sparse_probs.reset()
            parameters.reset(alpha, beta, mu, kappa)
        else:
            parameters.update_no_change(event, alpha, beta, mu, kappa)
        # Calculate probability of change point
        # attack_probs = parameters.priors(event)
        attack_probs = parameters.priors_cached(event, beta_cache)
        val_prob = change_point_algorithms.get_change_prob(attack_probs, sparse_probs)
        is_attack = val_prob <= 0.05
        yield is_attack


@profile
# @njit
def calculate_probabilities(
        event, alpha, beta, mu, kappa, run_lengths, probabilities, lamb,
        trunc_threshold=1e-16):
    """ """
    hazard = hazard_function(lamb)
    priors = np.empty_like(alpha)
    calculate_prior_arr_inplace(event, alpha, beta, mu, kappa, priors)
    new_probabilities = np.zeros(probabilities.size + 1)
    # here we define the type as uint32, this is arbitrary and might need to be changed later
    new_run_lengths = np.zeros(run_lengths.size + 1, dtype=np.uint32)
    # Multiply probabilities by their priors
    priors *= probabilities
    new_probabilities[1:] += priors
    # should be fine to multiply entire vector if first element is zero
    new_probabilities *= (1 - hazard)
    new_probabilities[0] += priors.sum()
    new_probabilities[0] *= hazard
    # Normalize probabilities
    if (prob_sum := new_probabilities.sum()) != 0.0:
        new_probabilities /= prob_sum
    # Match the run length values with the probabilities
    # new_run_lengths[0] = 0  # don't need this line since array initialized to zeros
    new_run_lengths[1:] += run_lengths
    new_run_lengths[1:] += 1
    # Truncate near zero values
    # trunc = new_probabilities < trunc_threshold
    # new_probabilities[trunc] = 0.0
    threshold_filter = new_probabilities > trunc_threshold
    threshold_filter[0] = True
    new_probabilities = new_probabilities[threshold_filter]
    new_run_lengths = new_run_lengths[threshold_filter]
    threshold_filter = threshold_filter[1:]
    new_alpha, new_beta, new_mu, new_kappa = alpha[threshold_filter], beta[threshold_filter], mu[threshold_filter], kappa[threshold_filter]
    # new_alpha, new_beta, new_mu, new_kappa = alpha, beta, mu, kappa
    return new_probabilities, new_alpha, new_beta, new_mu, new_kappa, new_run_lengths


@njit
def update_no_attack_arr(
        event: float, alpha_arr: np.ndarray, beta_arr: np.ndarray,
        mu_arr: np.ndarray, kappa_arr: np.ndarray, alpha: float, beta: float,
        mu: float, kappa: float):
    """ """
    size = alpha_arr.size + 1
    # update
    mu_p = np.empty(shape=size)
    kappa_p = np.empty(shape=size)
    alpha_p = np.empty(shape=size)
    beta_p = np.empty(shape=size)

    kappa_p[1:] = kappa_arr + 1
    alpha_p[1:] = alpha_arr + 0.5
    kappa_plus = kappa_arr + 1
    mu_p[1:] = (kappa_arr * mu_arr + event) / kappa_plus
    beta_p[1:] = beta_arr + kappa_arr * np.square(event - mu_arr) / (2 * kappa_plus)
    mu_p[0] = mu
    kappa_p[0] = kappa
    alpha_p[0] = alpha
    beta_p[0] = beta
    return alpha_p, beta_p, mu_p, kappa_p


@njit
def calculate_prior_arr(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF given parameters of inverse gamma distribution."""
    return t_func_arr(point, mus, ((betas * (kappas + 1.0)) / (alphas * kappas)), 2 * alphas)


@njit
def calculate_prior_arr_v1(point, alphas, betas, mus, kappas):
    """ Return student's T distribution PDF for given parameters of inverse gamma distribution."""
    t_values = calculate_prior_helper(point, alphas, betas, mus, kappas)
    t_values /= beta_numba(0.5, alphas)
    return t_values


@njit
def calculate_prior_helper(point, alphas, betas, mus, kappas):
    """ """
    denom = 2 * betas * (kappas + 1.0) / kappas
    t_values = (point - mus)**2 / denom
    t_values += 1.0
    # t_values **= -(alphas + 0.5)
    exponent = -(alphas + 0.5)
    t_values **= exponent
    t_values /= np.sqrt(denom)
    return t_values


# @profile
@njit
def calculate_prior_arr_inplace(point, alphas, betas, mus, kappas, out):
    """ """
    calculate_prior_helper_inplace(point, alphas, betas, mus, kappas, out)
    out /= beta_numba(0.5, alphas)
    # out /= scipy.special.beta(0.5, alphas)


@njit
def calculate_prior_helper_inplace(point, alphas, betas, mus, kappas, out):
    """ """
    arr = np.empty((2, alphas.size))
    denom = arr[0]
    exponent = arr[1]
    denom[:] = 2 * betas * (kappas + 1.0) / kappas
    out[:] = (point - mus)**2 / denom
    out += 1.0
    # t_values **= -(alphas + 0.5)
    exponent[:] = -(alphas + 0.5)
    out **= exponent
    out /= np.sqrt(denom)


def find_max_cp(probs):
    return np.argmax(probs)


# @njit
def hazard_function(lam: float):
    return 1 / lam


@njit
def t_func_arr(x_bar, mu_arr, s_arr, n_arr):
    """ """
    # t_values = np.zeros_like(mu_arr)
    s_n_arr = s_arr * n_arr
    n_half = n_arr * 0.5
    t_values = ((x_bar - mu_arr)**2 / s_n_arr + 1.0) ** (-(n_half + 0.5))

    t_values /= (np.sqrt(s_n_arr) * beta_numba(0.5, n_arr / 2))
    return t_values
    # old code
    # t_values = (x_bar - mu_arr) / np.sqrt(s_arr)
    # t_values = (1.0 + np.square(t_values) / n_arr) ** (-(n_arr + 1) / 2)
    # # t_values /= (np.sqrt(n_arr) * t_func_arr_helper_beta(n_arr)) * np.sqrt(s_arr)
    # t_values /= (np.sqrt(n_arr) * beta_numba(0.5, n_arr / 2)) * np.sqrt(s_arr)
    # # t_values /= (np.sqrt(n_arr) * scipy.special.beta(0.5, n_arr / 2)) * np.sqrt(s_arr)
    # return t_values
    # coeffs = (np.sqrt(dfs) * scipy.special.beta(0.5, dfs / 2))
    # return exponentials / (coeffs * np.sqrt(s_arr))


@vectorize(['float64(float64, float64)', 'float32(float32, float32)'], cache=False, nopython=True)
def beta_numba(val_1, val_2):
    """ Return vectorized function for """
    return math.exp(math.lgamma(val_1) + math.lgamma(val_2) - math.lgamma(val_1 + val_2))


def get_bocpd_from_generator(
        time, data, mu, kappa, alpha, beta, lamb, with_progress=False):
    """ """
    # Instantiating variables
    begin = 0
    # try to use rust version, if it's not in the wheel fall back to python implementation
    try:
        print('Start rust bocpd algorithm.')
        bocpd_model_gen = bocpd_rust_hybrid(data, mu, kappa, alpha, beta, lamb)
        if with_progress:
            shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
                time, begin, bocpd_model_gen, len(data))
        else:
            shocks, non_shocks = detection_to_intervals_for_generator_v1(
                time, begin, bocpd_model_gen)
    except NameError:
        print('Exception occurred, reverting to python')
        bocpd_model_gen = bocpd_generator(
                data, mu, kappa, alpha, beta, lamb)
        if with_progress:
            shocks, non_shocks = detection_to_intervals_for_generator_v1_with_progress(
                time, begin, bocpd_model_gen, len(data))
        else:
            shocks, non_shocks = detection_to_intervals_for_generator_v1(
                time, begin, bocpd_model_gen)
    return shocks, non_shocks
