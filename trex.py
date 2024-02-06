from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import torch as th
from tree_counters import TreeCounter, TreeCounterCV, TreeWrapper
import os
import numpy as np

class TREX:
    def __init__(
        self,
        env_id: str,
        on_policy_algorithm = PPO,
        count: bool = True, 
        normalize_env: bool = False,
        counter_cls = TreeCounter,
        counter_updt_freq: int = 16384,
        warm_start_only: bool = False,
        exploration_steps: int = np.inf
    ):
        self.env_id = env_id
        env = gym.make(self.env_id)
        log_dir = env_id + '-' + normalize_env * ('normalize' + '-')+ count * ("counting" + '-' + warm_start_only * ('only_wstrt' + '-')+ str(counter_updt_freq) + "-explo_stp" + str(exploration_steps))
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(env, log_dir)
        if normalize_env:
            # self.env = gym.wrappers.NormalizeReward(self.env)
            self.env = gym.wrappers.NormalizeObservation(self.env)
        # env = TreeWrapper(env, TreeCounter(), 16384)
        
        if count:
            self.tree_counter = counter_cls()
            self.counter_updt_freq = counter_updt_freq
            self.env = TreeWrapper(self.env, self.tree_counter, self.counter_updt_freq, only_warm_start=warm_start_only, exploration_steps=exploration_steps)
            self.warm_start = True
        else:
            self.warm_start = False

        self.agent = on_policy_algorithm("MlpPolicy", self.env, verbose=1)

    def do_warm_start(self):
        s = self.agent.get_env().reset()
        for _ in range(self.counter_updt_freq):
            action = self.agent.predict(th.FloatTensor(s), deterministic=False)[0]
            s, _, done, _ = self.agent.get_env().step(action)
            if done:
                s = self.agent.get_env().reset()

    def learn(self, total_timesteps:int = 100_000):
        if self.warm_start:
            self.do_warm_start()
        self.agent.learn(total_timesteps, progress_bar=True)
        

        

if __name__ == "__main__":
    trex = TREX("Swimmer-v4", normalize_env=True, count=True, exploration_steps=50_000)
    trex.learn()