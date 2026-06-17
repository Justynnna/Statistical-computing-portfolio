import pytest
from numstats.clt_poisson import poisson_pmf, poisson_extended, clt_interval_probability, calculate_sample_size, INF

def test_poisson_pmf():
    assert poisson_pmf(100, 0.02, 2) == pytest.approx(0.270670566)

def test_poisson_extended():
    result = poisson_extended(100, 0.02, [0, 1])
    assert result[0] == pytest.approx(0.4060058)
    assert result[1] == pytest.approx(0.04)

def test_clt_interval_probability():
    prob = clt_interval_probability(100, 0.5, 0.5, lower_bound=45, upper_bound=55)
    assert prob == pytest.approx(0.682689, abs=1e-4)

def test_calculate_sample_size():
    assert calculate_sample_size(0.5, 0.05, 0.95) == 385

