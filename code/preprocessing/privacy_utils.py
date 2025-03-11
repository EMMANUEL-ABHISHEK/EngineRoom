import numpy as np

def apply_differential_privacy(value: float, epsilon: float) -> float:
    """
    Applies Laplace noise to a given value based on the privacy parameter epsilon.
    
    Parameters:
    - value: The original numerical output (e.g., predicted engagement metric).
    - epsilon: The privacy budget (lower epsilon means higher noise).
    
    Returns:
    - The noisy value after applying Laplace noise.
    """
    scale = 1.0 / epsilon  # Scale factor for noise
    noise = np.random.laplace(0, scale)
    return value + noise

# Example usage:
original_engagement = 100  # e.g., 100 likes predicted
privacy_budget = 3  # Adjust this: 1 = max privacy (more noise), 10 = max accuracy (less noise)
noisy_engagement = apply_differential_privacy(original_engagement, privacy_budget)
print(f"Original: {original_engagement}, Noisy: {noisy_engagement}")
