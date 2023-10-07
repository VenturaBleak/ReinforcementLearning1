import math

class CosineAnnealingEpsilonScheduler:
    def __init__(self, start_epsilon, min_epsilon, total_episodes, restarts, decay_factor=0.9):
        self.start_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.total_episodes = total_episodes
        self.restarts = restarts
        self.decay_factor = decay_factor
        self.T_i = total_episodes // (restarts + 1)  # Episodes per restart
        self.episode_n = 0

    def step(self):
        self.episode_n += 1

        if self.episode_n % self.T_i == 0:  # Check for restarts
            self.start_epsilon *= self.decay_factor

        fraction = (self.episode_n % self.T_i) / self.T_i
        epsilon = self.min_epsilon + 0.5 * (self.start_epsilon - self.min_epsilon) * (1 + math.cos(math.pi * fraction))

        return epsilon

class PolynomialEpsilonDecay:
    """
    Polynomial epsilon decay until step reaches max_decay_steps.

    Args:
        start_epsilon: Starting value of epsilon (typically 1.0 for exploration-heavy start).
        end_epsilon: The minimum value to which epsilon can decay.
        max_decay_steps: After this step, epsilon value will be end_epsilon.
        power: The power of the polynomial.
    """

    def __init__(self, start_epsilon, end_epsilon, max_decay_steps, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.max_decay_steps = max_decay_steps
        self.power = power
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count > self.max_decay_steps:
            return self.end_epsilon
        return ((self.start_epsilon - self.end_epsilon) *
                (1 - self.step_count / self.max_decay_steps) ** self.power +
                self.end_epsilon)