import gymnasium as gym
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class ForestCounter:
    ### For prediction based exploration
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42, max_leaf_nodes=2**10, n_jobs=6, warm_start=True, n_estimators=8)
        self.dict_leaves = {}
        self.initialized = False
        self.is_fitted = False

    def update(self):
        self.fit()

    def fit(self):
        self.model.fit(
            np.concatenate((self.S, self.A), axis=1),
            self.Snext
        )
        self.model.n_estimators += 1
        self.is_fitted = True

    def count(self, s: np.ndarray, a: np.ndarray, sn: np.ndarray):
        bonus = np.linalg.norm(self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0] - sn)
        return  bonus
    
    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):
        if not self.initialized:
            self.S, self.A, self.Snext = (
                S.reshape(1, -1),
                A.reshape(1, -1),
                Snext.reshape(1, -1),
            )
            self.initialized = True
        else:
            self.S = np.concatenate((self.S, S.reshape(1, -1)), axis=0)
            self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
            self.Snext = np.concatenate((self.Snext, Snext.reshape(1, -1)), axis=0)


class TreeCounter:
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42, max_leaf_nodes=2**10)
        self.dict_leaves = {}
        self.initialized = False
        self.is_fitted = False

    def update(self):
        self.fit()

    def fit(self):
        self.model.fit(
            np.concatenate((self.S, self.A), axis=1),
            self.Snext
        )
        self.is_fitted = True

    def count(self, s: np.ndarray, a: np.ndarray):
        snext = self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0]
        snext_l = snext.tolist()
        snexttpl = tuple(snext_l)
        self.dict_leaves[snexttpl] = self.dict_leaves.get(snexttpl, 0) + 1
        # if self.dict_leaves[snexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[snexttpl]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):
        if not self.initialized:
            self.S, self.A, self.Snext = (
                S.reshape(1, -1),
                A.reshape(1, -1),
                Snext.reshape(1, -1),
            )
            self.initialized = True
        else:
            self.S = np.concatenate((self.S, S.reshape(1, -1)), axis=0)
            self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
            self.Snext = np.concatenate((self.Snext, Snext.reshape(1, -1)), axis=0)



class TreeCounterMiniGrid(TreeCounter):
    def __init__(self):
        super().__init__()

    def count(self, s: np.ndarray, a: np.ndarray):
        s = s.flatten()
        a = np.array([a])
        return super().count(s, a)

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):
        S = S.flatten()
        Snext = Snext.flatten()
        A = np.array([A])
        super().update_buffers(S, A, Snext)

class TreeCounterMiniGridWSOnly(TreeCounterMiniGrid):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeRegressor(random_state=42, max_leaf_nodes=2**14)

    def update(self):
        if self.is_fitted:
            return
        else:
            self.fit()

    def count(self, s: np.ndarray, a: np.ndarray):
        s = s.flatten()
        a = np.array([a])
        leaf = self.model.apply(np.concatenate((s, a)).reshape(1, -1))[0]
        self.dict_leaves[leaf] = self.dict_leaves.get(leaf, 0) + 1
        # if self.dict_leaves[snexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[leaf]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):  
        if self.is_fitted:
            return
        else:
            super().update_buffers(S, A, Snext)

class TreeCounterWSOnly(TreeCounter):
    def __init__(self):
        super().__init__()

    def update(self):
        if self.is_fitted:
            return
        else:
            self.fit()

    def count(self, s: np.ndarray, a: np.ndarray):
        leaf = self.model.apply(np.concatenate((s, a)).reshape(1, -1))[0]
        self.dict_leaves[leaf] = self.dict_leaves.get(leaf, 0) + 1
        # if self.dict_leaves[snexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[leaf]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):  
        if self.is_fitted:
            return
        else:
            super().update_buffers(S, A, Snext)



class TreeCounterCV:
    def __init__(self):
        self.model = GridSearchCV(
                DecisionTreeRegressor(random_state=42),
                param_grid={"max_leaf_nodes": 2**np.arange(7, 11)},
                n_jobs=6,
            )
        
        self.dict_leaves = {}
        self.initialized = False
        self.is_fitted = False

    def update(self):
        self.fit()

    def fit(self):
        self.model.fit(
            np.concatenate((self.S, self.A), axis=1),
            self.Snext
        )
        self.is_fitted = True

    def count(self, s: np.ndarray, a: np.ndarray):
        snext = self.model.best_estimator_.predict(
            np.concatenate((s, a)).reshape(1, -1)
        )[0]
        snext_l = snext.tolist()
        snexttpl = tuple(snext_l)
        self.dict_leaves[snexttpl] = self.dict_leaves.get(snexttpl, 0) + 1
        # if self.dict_leaves[snexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[snexttpl]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):
        if not self.initialized:
            self.S, self.A, self.Snext = (
                S.reshape(1, -1),
                A.reshape(1, -1),
                Snext.reshape(1, -1),
            )
            self.initialized = True
        else:
            self.S = np.concatenate((self.S, S.reshape(1, -1)), axis=0)
            self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
            self.Snext = np.concatenate((self.Snext, Snext.reshape(1, -1)), axis=0)


class TreeCounterCVWSOnly(TreeCounterCV):
    def __init__(self):
        super().__init__()

    def update(self):
        if self.is_fitted:
            return
        else:
            self.fit()

    def count(self, s: np.ndarray, a: np.ndarray):
        leaf = self.model.best_estimator_.apply(
            np.concatenate((s, a)).reshape(1, -1)
        )[0]
        self.dict_leaves[leaf] = self.dict_leaves.get(leaf, 0) + 1
        # if self.dict_leaves[snexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[leaf]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray
    ):
        if self.is_fitted:
            return
        else:
            super().update_buffers(S, A, Snext)
            


class TreeWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, tc: TreeCounter, update_tc_freq: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.current_state_tc = None
        self.tot_step_tc = 0
        self.tc = tc
        self.update_tc_freq = update_tc_freq

    def reset(self, **kwargs):
        self.current_state_tc, infos = self.env.reset()
        return self.current_state_tc, infos

    def step(self, action):
        snext, r, term, trunc, infos = self.env.step(action)
        if self.tc.is_fitted:
            if isinstance(self.tc, ForestCounter):
                bonus = self.tc.count(self.current_state_tc, action, snext)
            else:
                bonus = 1 / np.sqrt(self.tc.count(self.current_state_tc, action))
        else:
            bonus = 0
        self.tot_step_tc += 1

        self.tc.update_buffers(self.current_state_tc, action, snext)
        if (self.tot_step_tc % self.update_tc_freq == 0 ):
            self.tc.update()

        self.current_state_tc = snext
        reward = r + bonus
        return self.current_state_tc, reward, term, trunc, infos