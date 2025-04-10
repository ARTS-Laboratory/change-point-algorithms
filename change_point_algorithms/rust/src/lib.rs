use std::iter::zip;
use pyo3::prelude::*;
use dist_params::DistParams;
use beta_cache::BetaCache;
use sparse_probs::{SparseProb, SparseProbs};

mod dist_params;
mod beta_cache;
mod sparse_probs;
pub mod bocpd;

/// Updates the probability distribution for a set of T-distributions with observed point.
#[pyfunction]
fn calc_probabilities(point: f64, lamb: f64, params: &DistParams, probs: &mut SparseProbs) -> PyResult<()> {
    let hazard = lamb.recip();
    let priors = params.priors(point);
    probs.update_probs(priors, hazard)?;
    probs.normalize();
    Ok(())
}

/// Updates the probability distribution for a set of T-distributions with
/// observed point and cache for beta.
#[pyfunction]
fn calc_probabilities_cached(point: f64, lamb: f64, params: &DistParams, probs: &mut SparseProbs, cache: &mut BetaCache) -> PyResult<()> {
    let hazard = lamb.recip();
    let priors = params.priors_cached(point, cache);
    probs.update_probs(priors, hazard)?;
    probs.normalize();
    Ok(())
}

/// Truncate vectors
#[pyfunction]
fn truncate_vectors<'py>(threshold: f64, params: &mut DistParams, probs: &mut SparseProbs) -> usize {
    let threshold_filter: Vec<bool> = probs.iter().map(|prob| prob.get_value() >= threshold).collect();

    let mut tf_iter = threshold_filter.iter();
    params.retain_mut(|_| *tf_iter.next().unwrap());
    let mut tf_iter = threshold_filter.into_iter();
    probs.retain_mut(|_| tf_iter.next().unwrap());
    params.len()
}

#[pyfunction]
fn get_change_prob(priors: Vec<f64>, probs: &SparseProbs) -> f64 {
    zip(priors.iter(), probs.iter()).map(|(change, prob)| {
        change * prob.get_value()}).sum()
}

/// A Python module implemented in Rust.
#[pymodule]
fn change_point_algorithms(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_probabilities_cached, m)?)?;
    m.add_function(wrap_pyfunction!(calc_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(truncate_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(get_change_prob, m)?)?;
    m.add_class::<BetaCache>()?;
    m.add_class::<DistParams>()?;
    m.add_class::<SparseProb>()?;
    m.add_class::<SparseProbs>()?;
    Ok(())
}
