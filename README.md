# NumStats: Statistical Inference & Simulation Library

A comprehensive, production-ready Python library designed for core statistical computations, parameter estimation, hypothesis testing, and probabilistic simulations. This repository serves as a portfolio project demonstrating clean code practices, complete unit-testing coverage, and real-world data science applications.

## 🌟 Key Features
- **Parameter Estimation**: Point and interval estimation for population means, variances, and proportions using standard distributions (\(Z\), \(t\), \(\chi^2\)).
- **Hypothesis Testing**: Parametric checks for single or multiple groups (paired/independent) and \(\chi^2\) Goodness-of-Fit tests (Uniform and Poisson).
- **Central Limit Theorem (CLT) Simulations**: Experimental proof of the CLT using Poisson distributions to show convergence towards normality.
- **Production Standards**: Full type-hinting, Google-style docstrings, and robust exception handling.

---

## 📁 Repository Structure

```text
numstats-project/
├── numstats/                      # Main library package
│   ├── __init__.py
│   ├── clt_poisson.py             # CLT simulation logic and sampling distributions
│   ├── estimation.py              # Confidence intervals and variance point estimators
│   └── hypothesis_testing.py      # Statistical tests (Z-tests, t-tests, Chi-Square GoF)
├── tests/                         # Unit tests suite (pytest framework)
│   ├── test_clt_poisson.py
│   ├── test_estimation.py
│   └── test_hypothesis_testing.py
├── examples/                      # Interactive Jupyter Notebook guides
│   ├── clt_poisson.ipynb
│   ├── estimation.ipynb
│   └── hypothesis_testing.ipynb
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd numstats-project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---


## 📊 Modules & Quick Examples

### 1. Parameter Estimation (`estimation.py`)
Computes unbiased estimates and bounds under different sample size limits (\(n \le 30\) using Student's \(t\), \(n > 30\) using Normal approximation).

```python
from numstats.estimation import confidence_interval_mean_unknown_std

# Page load times monitored for 10 random website users (seconds)
load_times = [2.1, 2.5, 3.1, 1.9, 2.4, 2.8, 3.5, 2.2, 2.6, 2.9]
alpha = 0.05  # 95% Confidence Level

lower, upper = confidence_interval_mean_unknown_std(load_times, alpha)
print(f"We are 95% confident that the true average load time lies within: [{lower:.2f}s, {upper:.2f}s]")
```

### 2. Hypothesis Testing (`hypothesis_testing.py`)
Verifies claims against sample statistics using industry-standard alternative modes (`"less"`, `"greater"`, `"two-sided"`).

```python
from numstats.hypothesis_testing import independent_means_test

# A/B Testing user engagement metrics (Clicks per layout)
layout_a = [12, 15, 14, 11, 13]
layout_b = [19, 22, 21, 18, 20]

stat, fail_to_reject = independent_means_test(layout_a, layout_b, alpha=0.05)
if not fail_to_reject:
    print("Result: Reject H0. There is a statistically significant difference between layouts.")
```

### 3. CLT & Poisson Simulations (`clt_poisson.py`)
Demonstrates how the sample mean of independently drawn Poisson distributions collapses into a standard normal curve as sample groups scale up.

---

## 📓 Interactive Notebooks

Check out the `examples/` directory for interactive data insights, data visualizations (`matplotlib`), and case studies:
* Open `examples/estimation.ipynb` to see running calculations of stock volatilities and app conversions.
* Open `examples/hypothesis_testing.ipynb` to look at standard rejection regions and critical boundaries plots.
* Open `examples/clt_poisson.ipynb` to explore normal-distribution-based assumptions and estimate risk profiles.

---
## 🧪 Running the Test Suite

The project includes a robust test framework using `pytest` to guarantee absolute math precision and prevent regression issues (e.g., edge cases like `ZeroDivisionError` on zero-variance subsets).

To run all automated unit tests, simply execute:
```bash
pytest
```

To run a specific test suite with descriptive output:
```bash
pytest tests/test_hypothesis_testing.py -v
```

---

## 🧮 Requirements
- Python 3.14+
- NumPy
- SciPy
- Matplotlib
- Pytest (for development and test execution)
