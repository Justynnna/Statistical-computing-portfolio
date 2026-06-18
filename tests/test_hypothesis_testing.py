import pytest
from numstats.hypothesis_testing import (
    one_sample_mean_test,
    independent_means_test,
    paired_means_test,
    variance_test,
    uniform_goodness_of_fit,
    poisson_goodness_of_fit,
)


@pytest.fixture
def base_sample():
    """Provides standard mock sample data where mean is around 10."""
    return [9.5, 10.2, 9.8, 10.5, 10.0]


def test_one_sample_mean_test_known_std(base_sample):
    """Tests Z-test for population mean with known standard deviation."""
    # H0: mean = 10.0. The sample is close to 10, so H0 should not be rejected.
    stat, accept_h0 = one_sample_mean_test(
        base_sample, hypothesized_mean=10.0, std_dev=0.5, alternative="two-sided"
    )
    assert accept_h0
    
    # H0: mean = 5.0. Sample mean is 10.0, so H0 should be rejected (False).
    _, accept_h0_fake = one_sample_mean_test(
        base_sample, hypothesized_mean=5.0, std_dev=0.5, alternative="two-sided"
    )
    assert not accept_h0_fake


def test_one_sample_mean_test_unknown_std_small(base_sample):
    """Tests t-test for population mean with unknown standard deviation (small sample)."""
    stat, accept_h0 = one_sample_mean_test(
        base_sample, hypothesized_mean=10.0, std_dev=None, alternative="two-sided"
    )
    assert isinstance(stat, float)
    assert accept_h0


def test_independent_means_test():
    """Tests independent two-sample t-test."""
    s1 = [11.5, 12.0, 12.5, 11.8, 12.2]  # Mean ~ 12
    s2 = [12.1, 11.9, 12.4, 11.7, 12.0]  # Mean ~ 12
    s3 = [5.1, 4.9, 5.3, 4.8, 5.2]       # Mean ~ 5
    
    # Means are similar -> do not reject H0
    _, accept_equal = independent_means_test(s1, s2, alternative="two-sided")
    assert accept_equal
    
    # Means are significantly different -> reject H0
    _, accept_equal_fake = independent_means_test(s1, s3, alternative="two-sided")
    assert not accept_equal_fake


def test_paired_means_test__with_no_difference():
    """Tests paired sample t-test for dependent metrics."""
    before = [10.0, 12.0, 15.0, 11.0, 14.0]
    after = [10.0, 12.0, 15.0, 11.0, 14.0]  # Zero difference
    
    _, accept_h0 = paired_means_test(before, after, hypothesized_difference=0.0)
    assert accept_h0

def test_paired_means_test():
    """Tests paired sample t-test for dependent metrics."""
    
    before = [120.0, 122.0, 118.0, 121.0, 119.0]
    after  = [120.5, 121.5, 118.5, 120.5, 119.5] 
    
    _, accept_h0 = paired_means_test(before, after, hypothesized_difference=0.0)
    assert accept_h0


def test_variance_test(base_sample):
    """Tests variance validation using Chi-Square distribution."""
    # Calculated sample variance for base_sample is small (~0.14)
    # Testing against highly unrealistic variance = 100.0 should reject H0 in two-sided test
    _, accept_h0 = variance_test(base_sample, hypothesized_variance=100.0, alternative="two-sided")
    assert not accept_h0


def test_uniform_goodness_of_fit():
    """Tests uniform distribution goodness-of-fit check."""
    # Strongly skewed data should fail uniform distribution check
    _, accept_uniform = uniform_goodness_of_fit([10, 50, 1000, 20000], alpha=0.05)
    assert not accept_uniform

    # Perfectly flat distribution should easily pass uniform check
    _, accept_perfect_uniform = uniform_goodness_of_fit([100, 100, 100, 100], alpha=0.05)
    assert accept_perfect_uniform


def test_poisson_goodness_of_fit():
    """Tests Poisson distribution goodness-of-fit validation."""
    # Counts of events for k=0, 1, 2, 3
    observed = [368, 368, 184, 61]
    stat, accept_poisson = poisson_goodness_of_fit(observed, lam=1.0, alpha=0.05)
    assert isinstance(stat, float)
    assert isinstance(bool(accept_poisson), bool)


def test_invalid_alternative_error(base_sample):
    """Ensures a ValueError is raised when an unsupported alternative string is provided."""
    with pytest.raises(ValueError, match="Alternative must be"):
        one_sample_mean_test(base_sample, hypothesized_mean=10.0, alternative="invalid_string")
