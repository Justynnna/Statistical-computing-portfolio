import math
import pytest
from scipy.stats import norm, t, chi2

# Import functions from your main file (assuming it is named estimation.py)
from numstats.estimation import (
    sample_variance,
    confidence_interval_mean_known_std,
    confidence_interval_mean_unknown_std,
    confidence_interval_variance,
    confidence_interval_proportion,
)


@pytest.fixture
def sample_data():
    """Provides a consistent sample data array for test cases."""
    return [10.0, 12.0, 14.0, 16.0, 18.0]  # Mean = 14.0


def test_sample_variance(sample_data):
    """Verifies the correct calculation of unbiased sample variance."""
    # For this sample, the unbiased variance equals exactly 10.0
    assert math.isclose(sample_variance(sample_data), 10.0, rel_tol=1e-9)


def test_sample_variance_exception():
    """Ensures that a ValueError is raised when sample size is insufficient."""
    with pytest.raises(ValueError, match="Sample size must be greater than 1"):
        sample_variance([5.0])


def test_confidence_interval_mean_known_std(sample_data):
    """Tests the mean confidence interval with a known standard deviation (Z-method)."""
    alpha = 0.05
    std_dev = 3.0
    n = len(sample_data)
    mean = sum(sample_data) / n
    
    z_crit = norm.ppf(1 - alpha / 2)
    margin_of_error = z_crit * std_dev / math.sqrt(n)
    expected_lower = mean - margin_of_error
    expected_upper = mean + margin_of_error
    
    lower, upper = confidence_interval_mean_known_std(sample_data, std_dev, alpha)
    assert math.isclose(lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(upper, expected_upper, rel_tol=1e-9)


def test_confidence_interval_mean_unknown_std_small_sample(sample_data):
    """Tests the mean confidence interval for small samples (n <= 30, t-distribution)."""
    alpha = 0.05
    n = len(sample_data)
    mean = sum(sample_data) / n
    s_dev = math.sqrt(sample_variance(sample_data))
    
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_crit * s_dev / math.sqrt(n)
    expected_lower = mean - margin_of_error
    expected_upper = mean + margin_of_error
    
    lower, upper = confidence_interval_mean_unknown_std(sample_data, alpha)
    assert math.isclose(lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(upper, expected_upper, rel_tol=1e-9)


def test_confidence_interval_mean_unknown_std_large_sample():
    """Tests the mean confidence interval for large samples (n > 30, Z-distribution)."""
    alpha = 0.05
    # Generate a dummy large dataset (n = 35) with controlled variance
    large_data = [9.0] * 35
    large_data[-1] = 11.0
    
    n = len(large_data)
    mean = sum(large_data) / n
    s_dev = math.sqrt(sample_variance(large_data))
    
    z_crit = norm.ppf(1 - alpha / 2)
    margin_of_error = z_crit * s_dev / math.sqrt(n)
    expected_lower = mean - margin_of_error
    expected_upper = mean + margin_of_error
    
    lower, upper = confidence_interval_mean_unknown_std(large_data, alpha)
    assert math.isclose(lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(upper, expected_upper, rel_tol=1e-9)


def test_confidence_interval_variance(sample_data):
    """Tests the variance confidence interval using Chi-Square distribution."""
    alpha = 0.05
    n = len(sample_data)
    s_squared = sample_variance(sample_data)
    
    chi2_upper = chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_lower = chi2.ppf(alpha / 2, df=n - 1)
    expected_lower = (n - 1) * s_squared / chi2_upper
    expected_upper = (n - 1) * s_squared / chi2_lower
    
    lower, upper = confidence_interval_variance(sample_data, alpha)
    assert math.isclose(lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(upper, expected_upper, rel_tol=1e-9)


def test_confidence_interval_proportion():
    """Tests the population proportion confidence interval (Wald method)."""
    data = ["Yes", "Yes", "No", "Yes", "No"]  # n = 5, successes = 3, p = 0.6
    target = "Yes"
    alpha = 0.10
    
    p_hat = 0.6
    n = 5
    z_crit = norm.ppf(1 - alpha / 2)
    margin_of_error = z_crit * math.sqrt((p_hat * (1 - p_hat)) / n)
    expected_lower = p_hat - margin_of_error
    expected_upper = p_hat + margin_of_error
    
    lower, upper = confidence_interval_proportion(data, target, alpha)
    assert math.isclose(lower, expected_lower, rel_tol=1e-9)
    assert math.isclose(upper, expected_upper, rel_tol=1e-9)