import gymnasium as gym
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

class TreeCounter:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=14, random_state=42)
        self.dict_leaves = {}
        self.initialized = False
        self.is_fitted = False

    def update(self):
        self.fit()
        print("Prev dict size: {}".format(len(self.dict_leaves)))
        self.update_dict()
        print("New dict size: {}".format(len(self.dict_leaves)))
        print(
            "Score Tree: {}".format(
                self.model.score(
                    np.concatenate((self.S, self.A), axis=1),
                    np.concatenate((self.R, self.Snext), axis=1),
                )
            )
        )

    def fit(self):
        self.model.fit(
            np.concatenate((self.S, self.A), axis=1),
            np.concatenate((self.R, self.Snext), axis=1),
        )
        self.is_fitted = True

    def count(self, s: np.ndarray, a: np.ndarray):
        rsnext = self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0]
        rsnext_l = rsnext.tolist()
        rsnexttpl = tuple(rsnext_l)
        self.dict_leaves[rsnexttpl] += 1
        # if self.dict_leaves[rsnexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[rsnexttpl]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, R: np.ndarray, Snext: np.ndarray
    ):
        if not self.initialized:
            self.S, self.A, self.R, self.Snext = (
                S.reshape(1, -1),
                A.reshape(1, -1),
                np.array([R]).reshape(-1, 1),
                Snext.reshape(1, -1),
            )
            self.initialized = True
        else:
            self.S = np.concatenate((self.S, S.reshape(1, -1)), axis=0)
            self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
            self.R = np.concatenate((self.R, np.array([R]).reshape(-1, 1)), axis=0)
            self.Snext = np.concatenate((self.Snext, Snext.reshape(1, -1)), axis=0)

    def update_dict(self):
        n_nodes = self.model.tree_.node_count
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        values = self.model.tree_.value

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        tot_leaves = 0
        for i in range(n_nodes):
            if is_leaves[i]:
                tot_leaves += 1
                ksq = values[i].squeeze()
                k_l = ksq.tolist()
                ktpl = tuple(k_l)
                self.dict_leaves[ktpl] = self.dict_leaves.get(ktpl, 0)

        print("New Tree has {}".format(tot_leaves))


class TreeCounterCV:
    def __init__(self):
        self.model = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid={"max_depth": np.arange(7, 13)},
            n_jobs=6,
        )
        self.dict_leaves = {}
        self.initialized = False
        self.is_fitted = False

    def update(self):
        self.fit()
        print("Prev dict size: {}".format(len(self.dict_leaves)))
        self.update_dict()
        print("New dict size: {}".format(len(self.dict_leaves)))
        print("Score Tree: {}".format(self.model.best_score_))

    def fit(self):
        self.model.fit(
            np.concatenate((self.S, self.A), axis=1),
            np.concatenate((self.R, self.Snext), axis=1),
        )
        self.is_fitted = True

    def count(self, s: np.ndarray, a: np.ndarray):
        rsnext = self.model.best_estimator_.predict(
            np.concatenate((s, a)).reshape(1, -1)
        )[0]
        rsnext_l = rsnext.tolist()
        rsnexttpl = tuple(rsnext_l)
        self.dict_leaves[rsnexttpl] += 1
        # if self.dict_leaves[rsnexttpl] > 2:
        #     print("counting might work")
        return self.dict_leaves[rsnexttpl]

    def update_buffers(
        self, S: np.ndarray, A: np.ndarray, R: np.ndarray, Snext: np.ndarray
    ):
        if not self.initialized:
            self.S, self.A, self.R, self.Snext = (
                S.reshape(1, -1),
                A.reshape(1, -1),
                R.reshape(-1, 1),
                Snext.reshape(1, -1),
            )
            self.initialized = True
        else:
            self.S = np.concatenate((self.S, S.reshape(1, -1)), axis=0)
            self.A = np.concatenate((self.A, A.reshape(1, -1)), axis=0)
            self.R = np.concatenate((self.R, R.reshape(-1, 1)), axis=0)
            self.Snext = np.concatenate((self.Snext, Snext.reshape(1, -1)), axis=0)

    def update_dict(self):
        n_nodes = self.model.best_estimator_.tree_.node_count
        children_left = self.model.best_estimator_.tree_.children_left
        children_right = self.model.best_estimator_.tree_.children_right
        feature = self.model.best_estimator_.tree_.feature
        threshold = self.model.best_estimator_.tree_.threshold
        values = self.model.best_estimator_.tree_.value

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        tot_leaves = 0
        for i in range(n_nodes):
            if is_leaves[i]:
                tot_leaves += 1
                ksq = values[i].squeeze()
                k_l = ksq.tolist()
                ktpl = tuple(k_l)
                self.dict_leaves[ktpl] = self.dict_leaves.get(ktpl, 0)

        print("New Tree has {}".format(tot_leaves))


class TreeWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, tc: TreeCounter, update_tc_freq: int, only_warm_start: bool = False, exploration_steps: int=50_000):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.current_state_tc = None
        self.tot_step_tc = 0
        self.tc = tc
        self.update_tc_freq = update_tc_freq
        self.only_warm_start = only_warm_start
        self.updated_once = False
        self.explo_steps = exploration_steps

    def reset(self, **kwargs):
        self.current_state_tc, infos = self.env.reset()
        return self.current_state_tc, infos

    def step(self, action):
        snext, r, term, trunc, infos = self.env.step(action)
        if self.tc.is_fitted:
            bonus = 1 / np.sqrt(self.tc.count(self.current_state_tc, action))
            self.explo_steps - 1
        else:
            bonus = 0
        self.tot_step_tc += 1

        if not(self.only_warm_start and self.tc.is_fitted) and self.explo_steps > 0:
            self.tc.update_buffers(self.current_state_tc, action, r, snext)
            if (self.tot_step_tc % self.update_tc_freq == 0 ):
                self.tc.update()

        self.current_state_tc = snext
        reward = r + (self.explo_steps > 0) * bonus
        return self.current_state_tc, reward, term, trunc, infos