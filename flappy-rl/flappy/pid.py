
import numpy as np

class PIDAgent:
    def __init__(self, kp=17.43, kd=4.96, target_offset=0.12):
        """
        A simple PD controller for Flappy Bird.
        
        Args:
            kp: Proportional gain (response to position error).
            kd: Derivative gain (response to velocity).
            target_offset: Vertical offset from the center of the gap (normalized).
                           Positive values mean aiming below the center.
        """
        self.kp = kp
        self.kd = kd
        self.target_offset = target_offset

    def predict(self, observation, state=None, deterministic=True):
        """
        Predict the action given the observation.
        
        Args:
            observation: The observation from the environment.
                         Can be a single observation or a batch (vectorized).
                         If vectorized, we assume we only control the first env for now
                         or we handle batching.
            state: LSTM state (unused).
            deterministic: Unused.
            
        Returns:
            action: The selected action (0 or 1).
            state: None.
        """
        # Handle vectorized environments (batch dimension)
        if isinstance(observation, np.ndarray) and observation.ndim > 1:
            # If batch size > 1, we return a list/array of actions
            actions = [self._predict_single(obs) for obs in observation]
            return np.array(actions), None
        else:
            return self._predict_single(observation), None

    def _predict_single(self, obs):
        # Handle Frame Stacking:
        # If obs is stacked, it might be (n_stack, features) or (n_stack * features,)
        # We only care about the most recent frame.
        # Assuming VecFrameStack flattens or stacks.
        # Let's assume the standard structure: the last features are the most recent.
        # But wait, FlappyEnv obs_dim is small (~8).
        # If stacked 4 times, it's 32.
        # We need to extract the most recent 'bird_y', 'bird_vy', and 'gap_center'.
        
        # Let's try to infer structure.
        # If 1D array:
        #   features = obs_dim (unstacked)
        #   If len(obs) > features, it's stacked.
        #   We take the last 'features' elements.
        
        # Hardcoded indices from FlappyEnv:
        # 0: bird_y
        # 1: bird_vy
        # 2: dx_norm
        # 3: gap_center (if !use_rays)
        # 4: gap_height (if !use_rays)
        # ...
        
        # We assume use_rays=False for this PID (standard eval).
        # If use_rays=True, we can't easily find the gap center from rays without logic.
        
        # Let's assume the last chunk of the array corresponds to the current frame.
        # We need to know the single-frame obs dim.
        # Based on env.py, it's roughly 6-8 floats.
        # Let's assume we can find the gap center at index 3 relative to the start of the frame.
        
        # Actually, for a robust PID, we might just want to access the environment state directly
        # if we were cheating. But we should try to use the observation.
        
        # Let's assume the observation is NOT stacked for the core logic, 
        # or we slice the last N elements.
        # However, VecFrameStack usually stacks channels.
        # For 1D input, it creates (N_stack, D).
        # If the input to this function is (N_stack, D), we take obs[-1].
        
        curr_obs = obs
        if obs.ndim == 2: # (N_stack, D)
             curr_obs = obs[-1]
        elif obs.ndim == 1 and len(obs) > 10: # Likely flattened stacked obs
             # Heuristic: assume 4 stacks.
             chunk_size = len(obs) // 4
             curr_obs = obs[-chunk_size:]
             
        bird_y = curr_obs[0]
        bird_vy = curr_obs[1]
        # dx = curr_obs[2]
        gap_center = curr_obs[3]
        
        # Normalized coordinates: -1 (top) to +1 (bottom)
        
        # Error: Distance from Bird to Target
        # If Bird is at 0.5 and Gap is at 0.0 (higher up), Error = 0.5 - 0.0 = 0.5 (Positive)
        # We need to flap to go UP (decrease Y).
        # So Positive Error => Flap.
        
        target = gap_center + self.target_offset
        error = bird_y - target
        
        # PD Control
        # Output = Kp * error + Kd * (-velocity)
        # We want to oppose the velocity if we are closing in too fast?
        # Actually, simply:
        # We want upward acceleration if we are below target.
        # Flap provides upward impulse (negative velocity change).
        # Gravity provides downward acceleration.
        
        # Let's simplify:
        # We flap if we are below the target trajectory.
        # Predicted Y in next frames = bird_y + bird_vy * t
        # If Predicted Y > Target Y (Bird is below), Flap.
        
        # Heuristic PD:
        # signal = Kp * (bird_y - target) + Kd * bird_vy
        # If signal > Threshold, Flap.
        
        # bird_vy is normalized. Positive = Down.
        # If bird_vy is large positive (falling fast), we need to flap more.
        # So +Kd * bird_vy contributes to flapping.
        
        signal = self.kp * error + self.kd * bird_vy
        
        # Threshold: Flap if signal is strong enough.
        # Since flapping is discrete, we just threshold.
        if signal > 0.0:
            return 1
        return 0
