import numpy as np


class PIDAgent:
    def __init__(self, kp=17.43, kd=4.96, target_offset=0.12):

        self.kp = kp
        self.kd = kd
        self.target_offset = target_offset

    def predict(self, observation, state=None, deterministic=True):

        if isinstance(observation, np.ndarray) and observation.ndim > 1:

            actions = [self._predict_single(obs) for obs in observation]
            return np.array(actions), None
        else:
            return self._predict_single(observation), None

    def _predict_single(self, obs):

        curr_obs = obs
        if obs.ndim == 2:
            curr_obs = obs[-1]
        elif obs.ndim == 1 and len(obs) > 10:
            chunk_size = len(obs) // 4
            curr_obs = obs[-chunk_size:]

        bird_y = curr_obs[0]
        bird_vy = curr_obs[1]
        gap_center = curr_obs[3]

        target = gap_center + self.target_offset
        error = bird_y - target

        signal = self.kp * error + self.kd * bird_vy

        if signal > 0.0:
            return 1
        return 0
