from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import torch as th
from tree_counters import TreeCounter, TreeCounterCV, TreeWrapper, ForestCounter, TreeCounterCVWSOnly, TreeCounterWSOnly
import os
import numpy as np

class TREX:
    def __init__(
        self,
        env_id: str,
        on_policy_algorithm = PPO,
        counter_cls = None,
        counter_updt_freq: int = 16384,
    ):
        self.env_id = env_id
        env = gym.make(self.env_id)

        if counter_cls is not None:
            self.count = True
            log_dir = "ppo-default-params/" + env_id + '-' +  (counter_cls().__class__.__name__ +'-' + str(counter_updt_freq))
        else:
            self.count = False
            log_dir = "ppo-default-params/" + env_id

        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(env, log_dir)
            # self.env = gym.wrappers.NormalizeReward(self.env)
        self.env = gym.wrappers.NormalizeObservation(self.env)
        # env = TreeWrapper(env, TreeCounter(), 16384)
        if self.count:
            self.tree_counter = counter_cls()
            self.counter_updt_freq = counter_updt_freq
            self.env = TreeWrapper(self.env, self.tree_counter, self.counter_updt_freq)

        self.agent = on_policy_algorithm("MlpPolicy", self.env, verbose=1)

    def do_warm_start(self):
        s = self.agent.get_env().reset()
        for _ in range(self.counter_updt_freq):
            action = self.agent.predict(th.FloatTensor(s), deterministic=False)[0]
            s, _, done, _ = self.agent.get_env().step(action)
            if done:
                s = self.agent.get_env().reset()

    def learn(self, total_timesteps:int = 100_000):
        if self.count:
            total_timesteps -= self.counter_updt_freq # warm start already does some timesteps even though w/o learning
            self.do_warm_start()
        
        self.agent.learn(total_timesteps, progress_bar=True)
        

        

if __name__ == "__main__":
    # trex = TREX("MountainCarContinuous-v0") # PPO
    # trex.learn()


    trex = TREX("MountainCarContinuous-v0", counter_cls=ForestCounter, counter_updt_freq=2048)
    trex.learn()

    # trex = TREX("MountainCarContinuous-v0", counter_cls=TreeCounter)
    # trex.learn()

    # trex = TREX("MountainCarContinuous-v0", counter_cls=TreeCounterCV)
    # trex.learn()

    # trex = TREX("MountainCarContinuous-v0", counter_cls=TreeCounterWSOnly)
    # trex.learn()

    # trex = TREX("MountainCarContinuous-v0", counter_cls=TreeCounterCVWSOnly)
    # trex.learn()




