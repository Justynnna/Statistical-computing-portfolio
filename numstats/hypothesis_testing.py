import math
from typing import List, Tuple, Optional, Any
import numpy as np
from scipy.stats import norm, t, chi2, poisson


def one_sample_mean_test(
    sample: List[float],
    hypothesized_mean: float,
    std_dev: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Tuple[float, bool]:
    """Performs a one-sample Z-test or t-test for the population mean.

    Args:
        sample: A list of numerical sample values.
        hypothesized_mean: The mean value under the null hypothesis (H0).
        std_dev: Population standard deviation. If None, sample standard deviation is computed.
        alpha: The significance level.
        alternative: The alternative hypothesis type: "less", "greater", or "two-sided".

    Returns:
        A tuple containing (test_statistic, fail_to_reject_h0).
    """
    n = len(sample)
    sample_mean = sum(sample) / n
    distribution = norm
    dist_args: Tuple[Any, ...] = ()

    if std_dev is None:
        std_dev = math.sqrt(sum((x - sample_mean) ** 2 for x in sample) / (n - 1))
        if n <= 30:
            distribution = t
            dist_args = (n - 1,)

    test_stat = (sample_mean - hypothesized_mean) * math.sqrt(n) / std_dev

    if alternative == "less":
        critical_value = distribution.ppf(1 - alpha, *dist_args)
        return test_stat, test_stat > -critical_value
    elif alternative == "greater":
        critical_value = distribution.ppf(1 - alpha, *dist_args)
        return test_stat, test_stat < critical_value
    elif alternative == "two-sided":
        critical_value = distribution.ppf(1 - alpha / 2, *dist_args)
        return test_stat, -critical_value < test_stat < critical_value
    else:
        raise ValueError("Alternative must be 'less', 'greater', or 'two-sided'")


def independent_means_test(
    sample1: List[float],
    sample2: List[float],
    std_dev1: Optional[float] = None,
    std_dev2: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Tuple[float, bool]:
    """Performs a two-sample independent Z-test or t-test comparing two means.

    Uses a t-distribution with a pooled variance if population standard deviations are unknown.
    """
    n1, n2 = len(sample1), len(sample2)
    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2

    if std_dev1 is not None and std_dev2 is not None:
        test_stat = (mean1 - mean2) / math.sqrt((std_dev1 ** 2) / n1 + (std_dev2 ** 2) / n2)
        distribution = norm
        dist_args: Tuple[Any, ...] = ()
    else:
        pooled_variance = ((n1 - 1) * np.var(sample1, ddof=1) + (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2)
        std_error = math.sqrt(pooled_variance) * math.sqrt(1 / n1 + 1 / n2)
        test_stat = (mean1 - mean2) / std_error
        distribution = t
        dist_args = (n1 + n2 - 2,)

    critical_value = distribution.ppf(1 - alpha if alternative != "two-sided" else 1 - alpha / 2, *dist_args)

    if alternative == "less":
        return test_stat, test_stat > -critical_value
    elif alternative == "greater":
        return test_stat, test_stat < critical_value
    elif alternative == "two-sided":
        return test_stat, -critical_value < test_stat < critical_value
    else:
        raise ValueError("Alternative must be 'less', 'greater', or 'two-sided'")


def paired_means_test(
    sample1: List[float],
    sample2: List[float],
    hypothesized_difference: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Tuple[float, bool]:
    """Performs a paired sample t-test for dependent variables."""
    differences = [a - b for a, b in zip(sample1, sample2)]
    n = len(differences)
    mean_diff = sum(differences) / n
    std_dev_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in differences) / (n - 1))
    if std_dev_diff == 0:
        if mean_diff == hypothesized_difference:
            return 0.0, True
        else:
            return float('inf') if mean_diff > hypothesized_difference else float('-inf'), False
    test_stat = (mean_diff - hypothesized_difference) / std_dev_diff * math.sqrt(n)
    critical_value = t.ppf(1 - alpha if alternative != "two-sided" else 1 - alpha / 2, df=n - 1)

    if alternative == "less":
        return test_stat, test_stat > -critical_value
    elif alternative == "greater":
        return test_stat, test_stat < critical_value
    elif alternative == "two-sided":
        return test_stat, -critical_value < test_stat < critical_value
    else:
        raise ValueError("Alternative must be 'less', 'greater', or 'two-sided'")


def variance_test(
    sample: List[float],
    hypothesized_variance: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> Tuple[float, bool]:
    """Performs a Chi-Square test for a single population variance."""
    n = len(sample)
    sample_var = np.var(sample, ddof=1)
    test_stat = (n - 1) * sample_var / hypothesized_variance

    if alternative == "less":
        return test_stat, test_stat >= chi2.ppf(alpha, df=n - 1)
    elif alternative == "greater":
        return test_stat, test_stat <= chi2.ppf(1 - alpha, df=n - 1)
    elif alternative == "two-sided":
        return test_stat, chi2.ppf(alpha / 2, df=n - 1) <= test_stat <= chi2.ppf(1 - alpha / 2, df=n - 1)
    else:
        raise ValueError("Alternative must be 'less', 'greater', or 'two-sided'")


def uniform_goodness_of_fit(observed_counts: List[int], alpha: float = 0.05) -> Tuple[float, bool]:
    """Performs a Chi-Square goodness-of-fit test for a uniform distribution."""
    n = len(observed_counts)
    expected_count = sum(observed_counts) / n
    test_stat = sum(((obs - expected_count) ** 2) / expected_count for obs in observed_counts)
    critical_value = chi2.ppf(1 - alpha, df=n - 1)
    
    return test_stat, test_stat <= critical_value


def poisson_goodness_of_fit(observed_counts: List[int], lam: float, alpha: float = 0.05) -> Tuple[float, bool]:
    """Performs a Chi-Square goodness-of-fit test for a Poisson distribution."""
    n = len(observed_counts)
    total_elements = sum(observed_counts)
    test_stat = 0.0

    for i in range(n):
        prob = poisson.pmf(i, lam)
        expected_count = prob * total_elements
        if expected_count > 0:
            test_stat += ((observed_counts[i] - expected_count) ** 2) / expected_count

    critical_value = chi2.ppf(1 - alpha, df=n - 2)
    return test_stat, test_stat <= critical_value
