from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
from tree_counters import TreeCounter, TreeCounterCV, TreeWrapper


class TREX:
    def __init__(
        self,
        env_id: str,
        on_policy_algorithm = PPO,
        normalize_env: bool = False,
        counter_cls = TreeCounter,
        counter_updt_freq: int = 16384,
        warm_start: bool = True,
        warm_start_only: bool = False,
    ):
        self.env_id = env_id
        env = gym.make(self.env_id)
        if normalize_env:
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.NormalizeObservation(env)
        # env = TreeWrapper(env, TreeCounter(), 16384)
            
        self.tree_counter = counter_cls()
        self.counter_updt_freq = counter_updt_freq
        self.env = TreeWrapper(env, self.tree_counter, self.counter_updt_freq, only_warm_start=warm_start_only)
        self.agent = on_policy_algorithm("MlpPolicy", self.env)
        self.warm_start = warm_start

    def do_warm_start(self):
        s = self.agent.get_env().reset()
        for _ in range(self.counter_updt_freq):
            action = self.agent.predict(th.FloatTensor(s), deterministic=False)[0]
            s, _, done, _ = self.agent.get_env().step(action)
            if done:
                s = self.agent.get_env().reset()

    def learn(self, total_timesteps:int = 100_000):
        self.timesteps = total_timesteps
        if self.warm_start:
            self.do_warm_start()
        
        eval_env = Monitor(gym.make(self.env_id))
        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=total_timesteps//20,
            deterministic=True,
            render=False,
            verbose=1,
        )

        self.agent.learn(total_timesteps, callback=eval_callback, progress_bar=True)

        

if __name__ == "__main__":
    trex = TREX("Swimmer-v4")
    trex.learn()