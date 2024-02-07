##### Exploration and Model Learning Tricks following this [blog](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/):
- Normlalize Observation cf [RND-PPO](https://arxiv.org/abs/1810.12894) (*Implemented*).
- In practice, normalize rewards does not seem to work as running means do not incorporate well sparse high-rewards (*Implemented*). 
- Bisimilarity Measure [MICO](https://arxiv.org/pdf/2106.08229.pdf]) (two pairs $(s, a)$ should have a similar bisimiliratiy measure if they lead to similar $(r, s_{next})$) (*Not Implemented*).
- Non-episodic returns, cf [RND-PPO](https://arxiv.org/abs/1810.12894), $\texttt{env.reset()}$ is called only after a policy update is performed (*Not Implemented*).

##### Tree models and counters.
- Given a maximum depth $D$, tree can map any coninuous space to $2^D$ discete and countable leaves. Using [Decision Trees](https://scikit-learn.org/stable/modules/tree.html), one can learn:  $S \times A \rightarrow \{R \times S\}^{2^D}$, $S \times A \rightarrow \{S\}^{2^D}$, $S \times A \rightarrow \{R\}^{2^D}$
- One can see that trees can be used both to learn predictive models, and discrete counter when equipped with a dictionary memory where key are leaves labels and values are visitations of leaves.
- Tree can be used to do both Count-Based and Prediction Based $r^{i}_t = N_t(\text{TreeLeaf}(s_t, a_t))^{-\frac{1}{2}}$ or $r^{i}_t =\text{TreeScore}(s_t, a_t, r_t, s_{{next}_t})^{-\frac{1}{2}}$ or even combine both: $r^{i}_t = N_t(\text{TreeLeaf}(s_t, a_t))^{-\frac{1}{2}} \times \text{TreeScore}(s_t, a_t, r_t, s_{{next}_t})$
