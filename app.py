import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

st.title("Bayesian A/B/n Power Calculator (Control + 3 Variants)")

# --- Inputs ---
st.sidebar.header("Test Parameters")

p_A = st.sidebar.number_input("Baseline conversion rate (control)", 0.01, 0.99, 0.05, step=0.01, format="%.2f")

uplift_1 = st.sidebar.number_input("Expected uplift Variant B (%)", 0.0, 1.0, 0.10, step=0.01)
uplift_2 = st.sidebar.number_input("Expected uplift Variant C (%)", 0.0, 1.0, 0.15, step=0.01)
uplift_3 = st.sidebar.number_input("Expected uplift Variant D (%)", 0.0, 1.0, 0.20, step=0.01)

threshold = st.sidebar.slider("Posterior probability threshold", 0.5, 0.99, 0.95, step=0.01)
power_target = st.sidebar.slider("Target power", 0.5, 0.99, 0.8, step=0.01)

simulations = st.sidebar.slider("Simulations per step", 100, 2000, 500, step=100)
samples = st.sidebar.slider("Posterior samples", 1000, 10000, 3000, step=500)

# --- Simulation Function ---
def simulate_power_3variants(p_A, uplift_list, threshold, power_target, simulations, samples):
    alpha_prior = 1
    beta_prior = 1
    max_n = 500000
    step_n = 5000

    p_variants = [p_A * (1 + u) for u in uplift_list]
    powers = []

    n = 1000
    while n <= max_n:
        success_count = 0

        for _ in range(simulations):
            # Simulate conversion counts
            conv_control = np.random.binomial(n, p_A)
            conv_variants = [np.random.binomial(n, p) for p in p_variants]

            # Generate posterior samples
            control_samples = beta.rvs(alpha_prior + conv_control, beta_prior + n - conv_control, size=samples)
            variant_samples = [
                beta.rvs(alpha_prior + conv, beta_prior + n - conv, size=samples)
                for conv in conv_variants
            ]

            # Check if any variant beats control with P > threshold
            winner = any(
                (variant > control_samples).mean() > threshold
                for variant in variant_samples
            )

            if winner:
                success_count += 1

        power = success_count / simulations
        powers.append((n, power))

        if power >= power_target:
            break

        n += step_n

    return powers

# --- Run Simulation ---
uplift_list = [uplift_1, uplift_2, uplift_3]
results = simulate_power_3variants(p_A, uplift_list, threshold, power_target, simulations, samples)
sample_sizes, power_values = zip(*results)

# --- Output ---
if power_values[-1] >= power_target:
    st.success(f"✅ Minimum sample size per group: **{sample_sizes[-1]}**")
else:
    st.warning("⚠️ Did not reach target power within limits.")

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, power_values, marker='o')
plt.axhline(power_target, color='red', linestyle='--', label="Target Power")
plt.xlabel("Sample Size per Group")
plt.ylabel("Estimated Power")
plt.title("Power Curve: Control vs 3 Variants")
plt.grid(True)
plt.legend()
st.pyplot(plt)
