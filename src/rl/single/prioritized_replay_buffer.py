import numpy as np
import torch as th
from typing import Any, Dict, Optional, Tuple, Union, List
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces
from typing import NamedTuple

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor
    indices: np.ndarray

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer implementation based on "Prioritized Experience Replay" (Schaul et al. 2015).
    """

    def __init__(self,
                 buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = 'auto',
                 n_envs: int = 1,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 epsilon: float = 1e-6,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Initialize sum tree and min tree
        self.tree_capacity = 1
        while self.tree_capacity < self.buffer_size:
            self.tree_capacity *= 2
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, np.inf)
        self.max_priority = 1.0

    def _update_priority(self, idx: int, priority: float):
        """
        Update priority in the sum tree and min tree.

        :param idx: Index of the priority to update
        :param priority: New priority value
        """
        tree_idx = idx + self.tree_capacity - 1
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        self.max_priority = max(self.max_priority, priority)

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] = self.sum_tree[2 * tree_idx + 1] + self.sum_tree[2 * tree_idx + 2]
            self.min_tree[tree_idx] = min(self.min_tree[2 * tree_idx + 1], self.min_tree[2 * tree_idx + 2])

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add new experience to the buffer.
        """
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self._update_priority(idx, self.max_priority ** self.alpha)

    def sample(self, batch_size: int, beta: float = 0.4, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of samples to draw
        :param beta: Importance sampling weight exponent
        :param env: Optional environment for normalization
        :return: PrioritizedReplayBufferSamples containing the sampled experiences
        """
        indices = self._sample_proportional(batch_size)
        weights = self._calculate_weights(indices, beta)
        samples = self._get_samples(indices, env)
        return PrioritizedReplayBufferSamples(*samples, weights=th.tensor(weights, device=self.device), indices=indices)

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """
        Sample indices based on proportional priority.

        :param batch_size: Number of samples to draw
        :return: Sampled indices
        """
        indices = np.zeros(batch_size, dtype=np.int32)
        p_total = self.sum_tree[0]
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            indices[i] = self._retrieve(a + (b - a) * np.random.uniform(0, 1))

        return indices

    def _retrieve(self, upperbound: float) -> int:
        """
        Retrieve index based on priority.

        :param upperbound: Upper bound for the priority
        :return: Retrieved index
        """
        idx = 0
        while idx < self.tree_capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if upperbound <= self.sum_tree[left]:
                idx = left
            else:
                upperbound -= self.sum_tree[left]
                idx = right
        return idx - self.tree_capacity + 1

    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        """
        Calculate importance sampling weights.

        :param indices: Indices of the sampled experiences
        :param beta: Importance sampling weight exponent
        :return: Importance sampling weights
        """
        p_min = self.min_tree[0] / self.sum_tree[0]
        max_weight = (p_min * self.size()) ** -beta

        p_samples = self.sum_tree[indices + self.tree_capacity - 1] / self.sum_tree[0]
        weights = (p_samples * self.size()) ** -beta / max_weight
        return weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.

        :param indices: Indices of the sampled experiences
        :param priorities: New priorities
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            self._update_priority(idx, priority ** self.alpha)
