"""Unit tests for pure metric functions."""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_preset
from src.metrics import (
    copulas,
    descriptive,
    distributions,
    evt,
    linear_algebra,
    optimization,
    rmt,
    stochastic,
)


def test_summary_table_shape():
    prices = generate_preset("iid_normal", n_obs=200)
    rets = prices.pct_change().dropna()
    summary = descriptive.summary_table(rets)
    assert summary.shape[0] == 5
    assert summary.shape[1] == prices.shape[1]


def test_pca_variance_explained():
    prices = generate_preset("correlated_normal", n_obs=300)
    rets = np.log(prices / prices.shift(1)).dropna()
    res = linear_algebra.pca_analysis(rets, n_components=3)
    assert res.cumulative_variance[-1] <= 1.0 + 1e-9
    assert len(res.explained_variance_ratio) == 3


def test_ou_fit_recovers_theta():
    sim = stochastic.simulate_ou(0.0, 2.5, 0.0, 0.03, 2000, dt=1 / 252)
    params = stochastic.fit_ou(pd.Series(sim))
    assert 1.0 < params["theta"] < 4.5


def test_marchenko_pastur_bounds():
    lo, hi = rmt.marchenko_pastur_bounds(q=0.3)
    assert lo < hi
    assert lo > 0


def test_gpd_on_synthetic_tail():
    rng = np.random.default_rng(0)
    x = rng.pareto(2.0, 1000) + 1
    gpd = evt.fit_gpd_pot(x, quantile=0.9)
    assert gpd.n_exceedances > 10
    assert gpd.scale > 0


def test_student_t_beats_normal_on_fat_tails():
    rng = np.random.default_rng(1)
    x = rng.standard_t(4, 500) * 0.02
    norm = distributions.fit_normal(x)
    stud = distributions.fit_student_t(x)
    assert stud.aic < norm.aic


def test_markowitz_weights_sum_to_one():
    prices = generate_preset("iid_normal", n_obs=252)
    rets = prices.pct_change().dropna()
    mu = rets.mean().values * 252
    cov = rets.cov().values * 252
    w = optimization.min_variance_portfolio(mu, cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert np.all(w >= -1e-6)


def test_svd_reconstruction_error_decreases_with_rank():
    prices = generate_preset("correlated_normal", n_obs=200)
    rets = np.log(prices / prices.shift(1)).dropna()
    low = linear_algebra.svd_analysis(rets, rank=2).reconstruction_error
    high = linear_algebra.svd_analysis(rets, rank=8).reconstruction_error
    assert high <= low + 1e-9


def test_svd_error_curve_monotone():
    prices = generate_preset("correlated_normal", n_obs=200)
    rets = np.log(prices / prices.shift(1)).dropna()
    _s, errs = linear_algebra.svd_relative_error_curve(rets)
    assert len(errs) >= 2
    assert np.all(np.diff(errs) <= 1e-9)


def test_ledoit_wolf_correlation():
    prices = generate_preset("iid_normal", n_obs=300)
    rets = prices.pct_change().dropna()
    _cov, corr = descriptive.ledoit_wolf_covariance(rets)
    eig = np.linalg.eigvalsh(corr.values)
    assert np.min(eig) > -1e-8


def test_ols_matches_mse():
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(100), rng.normal(size=(100, 2))])
    beta = np.array([0.5, -0.2, 0.9])
    y = X @ beta + rng.normal(0, 0.05, 100)
    b = optimization.ols_closed_form(X, y)
    mse_ols = optimization.linear_mse_loss(b, X, y)
    mse_random = optimization.linear_mse_loss(rng.normal(size=3), X, y)
    assert mse_ols < mse_random


def test_pca_svd_agreement_tight():
    prices = generate_preset("correlated_normal", n_obs=400)
    rets = np.log(prices / prices.shift(1)).dropna()
    _, _, d = linear_algebra.pca_svd_agreement(rets, standardize=True, n_components=min(5, rets.shape[1] - 1))
    assert d < 1e-10


def test_markowitz_analytic_matches_numerical_unconstrained():
    rng = np.random.default_rng(11)
    n = 6
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n + np.eye(n) * 0.05
    mu = rng.normal(scale=0.12, size=n)
    frontier = optimization.markowitz_frontier(mu, cov, n_points=30, long_only=False)
    assert frontier
    p = frontier[len(frontier) // 2]
    tgt = float(p.target_return)
    w_a = optimization.markowitz_min_variance_target_analytic(mu, cov, tgt)
    assert abs(float(w_a @ mu) - tgt) < 1e-8
    v_num = float(np.sqrt(max(p.weights @ cov @ p.weights, 0.0)))
    v_ana = float(np.sqrt(max(w_a @ cov @ w_a, 0.0)))
    assert abs(v_num - v_ana) < 1e-4


def test_gmv_closed_form_normalized():
    rng = np.random.default_rng(3)
    n = 5
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n + 0.05 * np.eye(n)
    w = optimization.global_minimum_variance_weights_analytic(cov)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)
    w2 = optimization.min_variance_portfolio(np.zeros(n), cov, long_only=False)
    v_a = float(w @ cov @ w)
    v_n = float(w2 @ cov @ w2)
    assert abs(v_a - v_n) < 1e-5


def test_tangency_portfolio_normalized():
    rng = np.random.default_rng(4)
    n = 4
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n + 0.08 * np.eye(n)
    mu = rng.normal(scale=0.15, size=n)
    w = optimization.tangency_portfolio_weights_uncstr(mu, cov, rf=0.02)
    assert np.isclose(np.sum(w), 1.0, atol=1e-9)


def test_gaussian_conditional_matches_linear_projection():
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    y = 0.3 * x + rng.normal(scale=0.5, size=500)
    g = distributions.gaussian_bivariate_moments(x, y)
    xm = 0.2
    m1, _ = distributions.conditional_gaussian_mean_var_y_given_x(xm, g)
    m2 = g.mean_y + g.beta_y_on_x * (xm - g.mean_x)
    assert abs(m1 - m2) < 1e-12


def test_linear_mse_grad_at_ols_near_zero():
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 1))])
    beta = np.array([0.4, -0.7])
    y = X @ beta + rng.normal(0, 0.08, size=50)
    w_hat = optimization.ols_closed_form(X, y)
    g = optimization.linear_mse_grad(w_hat, X, y)
    assert np.max(np.abs(g)) < 1e-5


def test_brownian_quadratic_variation_ratio():
    paths = stochastic.simulate_brownian(500, n_paths=3)
    stats = stochastic.brownian_increment_stats(paths)
    assert 0.7 < stats["qv_ratio"] < 1.3


def test_ito_log_correction_negative():
    info = stochastic.apply_ito_lemma_gbm_log(0.08, 0.25, 1.0)
    assert info["correction"] < 0


def test_gpd_var_increases_when_threshold_lowers():
    rng = np.random.default_rng(2)
    x = rng.pareto(2.5, 800) + 1
    high_u = float(np.quantile(x, 0.95))
    low_u = float(np.quantile(x, 0.90))
    g_high = evt.fit_gpd_pot(x, threshold=high_u)
    g_low = evt.fit_gpd_pot(x, threshold=low_u)
    assert g_low.var_99 >= g_high.var_99 * 0.5


def test_hill_positive_on_pareto():
    rng = np.random.default_rng(9)
    x = rng.pareto(2.0, 500) + 1
    xi, _ks, _hill = evt.hill_tail_index(x, k=50)
    assert xi > 0


def test_clayton_kendall_tau_roundtrip():
    samples = copulas.clayton_samples(2.5, 500, seed=10)
    u, v = samples[:, 0], samples[:, 1]
    x = copulas.fit_clayton_copula(u * 10, v * 10)
    assert x.parameter is not None
    assert x.parameter > 0


def test_clayton_tail_dependence_positive():
    tail = copulas.tail_dependence_coefficients("clayton", theta=2.0)
    assert tail.lambda_lower > 0


def test_rmt_denoise_correlation_psd():
    prices = generate_preset("crash_correlation", n_obs=400)
    rets = prices.pct_change().dropna()
    corr_clean, _, _ = rmt.denoise_correlation(rets, method="constant")
    eig = np.linalg.eigvalsh(corr_clean.values)
    assert np.min(eig) > -1e-6


def test_gbm_euler_matches_direct_sim():
    s_direct = stochastic.simulate_gbm(100.0, 0.1, 0.2, 100, n_paths=1, seed=99)[:, 0]
    s_euler = stochastic.simulate_gbm_euler(100.0, 0.1, 0.2, 100, seed=99)
    assert np.allclose(s_direct, s_euler, rtol=0.05)


def test_ou_mle_matches_ar1_fit():
    sim = stochastic.simulate_ou(0.0, 3.0, 0.0, 0.04, 1500, dt=1 / 252)
    s = pd.Series(sim)
    ar = stochastic.fit_ou(s)
    mle = stochastic.ou_max_likelihood(s)
    assert abs(ar["theta"] - mle["theta"]) < 1e-10


def test_frank_copula_negative_tau():
    rng = np.random.default_rng(12)
    a = rng.normal(size=400)
    b = -a + rng.normal(scale=0.4, size=400)
    res = copulas.fit_frank_copula(a, b)
    assert res.parameter is not None
    assert res.parameter < 0


def test_gpd_var_at_confidence_matches_pot():
    rng = np.random.default_rng(13)
    x = rng.pareto(2.2, 600) + 1
    u = float(np.quantile(x, 0.92))
    gpd = evt.fit_gpd_pot(x, threshold=u)
    v99, _e99 = evt.gpd_var_es_at_confidence(
        x, gpd.threshold, gpd.shape, gpd.scale, gpd.n_exceedances, 0.99
    )
    assert abs(v99 - gpd.var_99) < 1e-6


def test_mp_bulk_majority_for_iid_noise():
    rng = np.random.default_rng(14)
    n_obs, n_assets = 400, 40
    r = rng.normal(0, 0.01, size=(n_obs, n_assets))
    df = pd.DataFrame(r, columns=[f"A{i}" for i in range(n_assets)])
    evals = rmt.correlation_eigenvalues(df)
    q = n_assets / n_obs
    lo, hi = rmt.marchenko_pastur_bounds(q)
    in_bulk = np.sum((evals >= lo) & (evals <= hi))
    assert in_bulk >= 0.5 * len(evals)


def test_summary_table_rejects_empty_frame():
    with pytest.raises(ValueError):
        descriptive.summary_table(pd.DataFrame())


def test_correlation_rejects_zero_variance_columns():
    df = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError):
        descriptive.correlation_matrix(df)


def test_joint_binning_rejects_invalid_bins():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        distributions.joint_binning(x, y, bins=1)


def test_clt_standardized_requires_positive_n():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        distributions.clt_standardized_sample_means(x, n=0)


def test_gbm_calibration_requires_positive_prices():
    s = pd.Series([100.0, 0.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        stochastic.calibrate_gbm(s)


def test_ou_fit_rejects_non_mean_reverting_series():
    series = pd.Series(np.linspace(0.0, 1.0, 100))
    with pytest.raises(ValueError):
        stochastic.ou_max_likelihood(series)


def test_evt_es_nan_when_shape_gte_one():
    x = np.linspace(1.0, 10.0, 100)
    var_level, es_level = evt.gpd_var_es_at_confidence(
        x,
        threshold=2.0,
        shape=1.2,
        scale=1.0,
        n_exceedances=20,
        confidence=0.99,
    )
    assert np.isfinite(var_level)
    assert np.isnan(es_level)


def test_denoise_rejects_under_sampled_panel():
    rng = np.random.default_rng(33)
    df = pd.DataFrame(rng.normal(size=(20, 40)))
    with pytest.raises(ValueError):
        rmt.denoise_correlation(df)


def test_marchenko_pastur_rejects_bad_q():
    with pytest.raises(ValueError):
        rmt.marchenko_pastur_bounds(1.0)


def test_markowitz_frontier_rejects_bad_cov_shape():
    mu = np.array([0.1, 0.12])
    cov = np.array([[1.0, 0.0, 0.1], [0.0, 1.0, 0.1]])
    with pytest.raises(ValueError):
        optimization.markowitz_frontier(mu, cov, n_points=3, long_only=False)
