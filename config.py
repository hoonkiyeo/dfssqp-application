"""Application settings. Edit the values here before running an experiment.

Starting point: the agreed October 9 baseline, with iota1 = 1.
This file only defines settings; importing it does not train or run anything.
"""

from math import sqrt


# Data: monthly observations built from daily industry returns.
dimension = 48
returns_file = "data/raw/48_Industry_Portfolios_daily_TXT.zip"
calendar_file = "data/raw/F-F_Research_Data_Factors_daily_TXT.zip"
returns_sha256 = "86e86a29e4365c13974631972d509f8ed6406e47bddf6eb90453fe19b95fb7e1"
calendar_sha256 = "2aea744a2d55a2d8fcccc8815aedef587fb16bdc13875c459f022963ca0a9992"
data_vintage = "2026-09-10"
portfolio_weighting = "value"
benchmark_ddof = 1  # Sample SD for the training-fitted inverse-volatility benchmark.
training_months = ("1969-07", "2003-09")
validation_months = ("2003-10", "2015-02")
test_months = ("2015-03", "2026-07")  # Reserved; not used for development.


# DF-SSQP schedules, for k = 0, 1, ...
# alpha_k = iota1 / (k + 1)**p1       step schedule
# beta_k  = iota2 / (k + 1)**p2       averaging schedule
# b_k     = iota3 / (k + 1)**p3       probe schedule
iota1 = 1.0
iota2 = 1.0
iota3 = 1.0 / sqrt(dimension)
p1 = 0.751
p2 = 0.501
p3 = 0.25


# DF-SSQP: first-order B = I; independent +/-1 perturbation coordinates.
sigma = 0.1
epsilon = 0.1
L_f = 1.0  # Baseline input to the step rule, not a measured Lipschitz bound.
L_c = 2.0  # Exact for c(w) = w.T @ w - 1.
jacobian_lower = 1e-3
jacobian_upper = 1e3
psi = 0.0  # Selects the lower endpoint of the actual-step interval.
p = 1.5    # Inactive when psi = 0.


# Initialization: one normalized Gaussian draw shared across all runs and modes.
# w0 = v / ||v||, v drawn using initialization_seed.
initialization_seed = 16_100_901
lambda_initial = 0.0
gradient_average_initial = 0.0  # Every coordinate of g_bar_{-1}.
jacobian_average_initial = 0.0  # Every entry of G_bar_{-1}.
tau_initial = 1.0
nu_initial = 1.0


# Inner model: beta*u + F_theta(w.T @ z), with a 1-8-1 neural link.
hidden_width = 8
activation = "softplus"
initialization_gain = 1.0  # Xavier-uniform weights; all neural biases start at zero.
neural_bias_initial = 0.0
linear_beta_initial = 0.0

# Inner training: fresh model at each loss evaluation, fixed w.
learning_rate = 0.01
training_steps = 1_000
batch_size = 64
adam_betas = (0.9, 0.999)
adam_epsilon = 1e-8
minibatch_policy = "reshuffle"
standardize = True  # Index, control and response; statistics from training only.
standardization_ddof = 0
dtype = "float64"
device = "cpu"
torch_threads = 1


# Experiment: ten stochastic runs, each with 5,000 SQP steps in AI and ZI.
# AI uses the exact constraint Jacobian; ZI estimates it.
modes = ("AI", "ZI")
sqp_steps = 5_000
outer_seeds = list(range(56_100_901, 56_100_911))
# Each outer seed determines that run's perturbations and training seeds.
# Both sides of a probe pair use the same training seed with fresh models.

# Independent evaluation panel: these seeds do not select or update w.
evaluation_seeds = list(range(46_100_901, 46_100_921))
evaluation_steps = [
    0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
]
baselines = ("aggregate_only", "mean", "pca")
experiment_name = "baseline"
runs_directory = "runs"
