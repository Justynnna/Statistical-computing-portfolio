import math
from typing import List, Tuple, Any
from scipy.stats import norm, t, chi2


def sample_variance(data: List[float]) -> float:
    """Calculates the unbiased sample variance (Bessel's correction).

    Args:
        data: A list of numerical sample values.

    Returns:
        The unbiased point estimate of the population variance.
    """
    n = len(data)
    if n <= 1:
        raise ValueError("Sample size must be greater than 1 to calculate variance.")
        
    mean = sum(data) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    
    return squared_diff_sum / (n - 1)


def confidence_interval_mean_known_std(
    data: List[float], std_dev: float, alpha: float
) -> Tuple[float, float]:
    """Calculates the confidence interval for the population mean 
    when the population standard deviation is known (Z-test method).

    Args:
        data: A list of numerical sample values.
        std_dev: The known population standard deviation.
        alpha: The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    mean = sum(data) / n
    
    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * std_dev / math.sqrt(n)
    
    return float(mean - margin_of_error), float(mean + margin_of_error)


def confidence_interval_mean_unknown_std(
    data: List[float], alpha: float
) -> Tuple[float, float]:
    """Calculates the confidence interval for the population mean 
    when the population standard deviation is unknown. 
    
    Uses Student's t-distribution for small samples (n <= 30) 
    and Normal distribution for large samples (n > 30).

    Args:
        data: A list of numerical sample values.
        alpha: The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    mean = sum(data) / n
    s_dev = math.sqrt(sample_variance(data))
    
    if n <= 30:
        critical_value = t.ppf(1 - alpha / 2, df=n - 1)
    else:
        critical_value = norm.ppf(1 - alpha / 2)
        
    margin_of_error = critical_value * s_dev / math.sqrt(n)
    
    return float(mean - margin_of_error), float(mean + margin_of_error)


def confidence_interval_variance(
    data: List[float], alpha: float
) -> Tuple[float, float]:
    """Calculates the confidence interval for the population variance 
    using the Chi-Square distribution.

    Args:
        data: A list of numerical sample values.
        alpha: The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    s_squared = sample_variance(data)
    
    chi2_upper = chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_lower = chi2.ppf(alpha / 2, df=n - 1)
    
    lower_bound = (n - 1) * s_squared / chi2_upper
    upper_bound = (n - 1) * s_squared / chi2_lower
    
    return float(lower_bound), float(upper_bound)


def confidence_interval_proportion(
    data: List[Any], target_value: Any, alpha: float
) -> Tuple[float, float]:
    """Calculates the confidence interval for a population proportion 
    (Wald confidence interval).

    Args:
        data: A list of sample observations (can be numbers, strings, etc.).
        target_value: The specific value/success to estimate the proportion for.
        alpha: The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    proportion = data.count(target_value) / n
    
    z_critical = norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * math.sqrt((proportion * (1 - proportion)) / n)
    
    return float(proportion - margin_of_error), float(proportion + margin_of_error)
