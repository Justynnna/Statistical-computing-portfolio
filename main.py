from numstats.clt_poisson import poisson_pmf, poisson_extended, clt_interval_probability, calculate_sample_size, INF
import math
n = 1000
p = 0.4
mean = 0.4
std_dev = math.sqrt(0.4 * 0.6)  # ok. 0.4898

# Szukamy prawdopodobieństwa w przedziale [370, 430]
prob = clt_interval_probability(n, mean, std_dev, lower_bound=370, upper_bound=430)
print(f"Szansa, że trafimy w przedział przy n=1000: {prob:.4f}")
wymagane_n = calculate_sample_size(p=0.4, margin_of_error=0.03, confidence_level=prob)
print(f"Minimalne wymagane n: {wymagane_n}")
