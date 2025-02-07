"""
Utility functions for testing.
"""


def get_accurate_observation(algorithm):
    """
    This function takes an algorithm, which has an environment.
    It should return an "observation" which can be used to mock the step/reset function.
    """
    combined_states = algorithm.environment._combine_states()
    return combined_states
