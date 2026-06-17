from scipy.stats import norm
from math import exp, sqrt, factorial, ceil, pow

INF = float('inf')

def poisson_pmf(n, p, k):
    """Calculate the Probability Mass Function (PMF) of a Poisson distribution.

    Args:
        n (int): Number of trials.
        p (float): Probability of success in a single trial.
        k (int): Number of occurrences (successes).

    Returns:
        float: The probability of exactly k successes.
    """
    lam = n * p
    result = pow(lam, k) * exp(-lam) / factorial(k)
    return result

def poisson_extended(n, p, k_list):
    """Calculate the cumulative probability for a list of successes and the variance.

    Args:
        n (int): Number of trials.
        p (float): Probability of success in a single trial.
        k_list (list of int): List of target success counts to aggregate.

    Returns:
        list: A list containing [total_probability, variance_per_trial].
    """
    total_prob = 0
    lam = p * n
    for k in k_list:
        total_prob += poisson_pmf(n, p, k)
    result = []
    result.append(total_prob)
    result.append(pow(lam, 2) / n)
    return result

def clt_interval_probability(n, mean, std_dev, lower_bound=-INF, upper_bound=INF):
    """Estimate the probability that the sum of n independent random variables 
    falls within a specified interval using the Central Limit Theorem (CLT).

    Args:
        n (int): Number of independent random variables.
        mean (float): Expected value (mean) of a single random variable.
        std_dev (float): Standard deviation of a single random variable.
        lower_bound (float, optional): Lower limit of the interval. Defaults to -INF.
        upper_bound (float, optional): Upper limit of the interval. Defaults to INF.

    Returns:
        float: Estimated probability that the sum falls between lower and upper bounds.
    """
    z_upper = (upper_bound - n * mean) / (std_dev * sqrt(n))
    z_lower = (lower_bound - n * mean) / (std_dev * sqrt(n))
    result = norm.cdf(z_upper) - norm.cdf(z_lower)
    return result

def calculate_sample_size(p, margin_of_error, confidence_level):
    """Calculate the minimum sample size required to estimate a population proportion.

    Args:
        p (float): Estimated proportion of successes (bound on probability).
        margin_of_error (float): Maximum acceptable margin of error (d).
        confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).

    Returns:
        int: The required sample size rounded up to the nearest integer.
    """
    z_score = norm.ppf((1 + confidence_level) / 2)
    numerator= p * (1 - p) * z_score ** 2
    denominator = margin_of_error * margin_of_error
    result = numerator / denominator
    return ceil(result)