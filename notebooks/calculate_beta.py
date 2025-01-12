# %%
import numpy as np


def sigma_rel_to_gamma(sigma_rel):
    # Calculate gamma from sigma_rel using the formula from the paper
    t = sigma_rel**-2
    # Solve the cubic equation to find gamma
    roots = np.roots([1, 7, 16 - t, 12 - t])
    # Return the maximum real root
    return roots.real.max()


def calculate_beta(sigma_rel, t):
    # Convert sigma_rel to gamma
    gamma = sigma_rel_to_gamma(sigma_rel)
    # Calculate beta for a given training step t
    beta = (1 - 1 / t) ** (gamma + 1)
    return beta


# Example usage
sigma_rel_05 = 0.05
sigma_rel_10 = 0.10
training_step = 10000  # Example training step

beta_05 = calculate_beta(sigma_rel_05, training_step)
beta_10 = calculate_beta(sigma_rel_10, training_step)

print(f"Beta for sigma_rel 0.05 at step {training_step}: {beta_05}")
print(f"Beta for sigma_rel 0.10 at step {training_step}: {beta_10}")
# %%

import numpy as np


def beta_to_gamma(beta, t):
    # Calculate gamma from beta
    gamma_plus_1 = np.log(beta) / np.log(1 - 1 / t)
    return gamma_plus_1 - 1


def gamma_to_sigma_rel(gamma):
    # Calculate sigma_rel from gamma
    return np.sqrt((gamma + 1) / ((gamma + 2) * (gamma + 3)))


def calculate_sigma_rel_for_beta(beta, t=10000):
    # Calculate gamma for the given beta
    gamma = beta_to_gamma(beta, t)
    # Calculate sigma_rel from gamma
    sigma_rel = gamma_to_sigma_rel(gamma)
    return sigma_rel


# Example usage
beta = 0.95
sigma_rel = calculate_sigma_rel_for_beta(beta)

print(f"Sigma_rel for beta {beta}: {sigma_rel}")
# %%
