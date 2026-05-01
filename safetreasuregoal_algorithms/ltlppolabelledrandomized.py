import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from safetreasuregoal_randomized import SafeTreasureDoorKeyGrid


class LabelledLTLSafeTreasureWrapper:
    AP = ["key", "door", "goal", "unsafe"]
    N_Q = 4

    def __init__(self, env, progress_bonus=0.5, cliff_penalty=0.0):
        self.env = env
        self.action_space = env.action_space
        self.progress_bonus = progress_bonus
        self.cliff_penalty = cliff_penalty
        self.q = 0
        self.n_q = self.N_Q
        self.n_ap = len(self.AP)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.q = 0
        labels = self.label(obs, info)
        info["labels"] = labels
        info["label_vector"] = self.label_vector(labels)
        info["ltl_q"] = self.q
        return self.flatten(obs, labels), info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        labels = self.label(obs, info)
        old_q = self.q
        new_q, ltl_bonus = self.dfa_transition(self.q, labels)

        self.q = new_q
        reward = env_reward + ltl_bonus

        info["labels"] = labels
        info["label_vector"] = self.label_vector(labels)
        info["ltl_q"] = self.q
        info["ltl_old_q"] = old_q
        info["ltl_bonus"] = ltl_bonus
        info["env_reward"] = env_reward
        info["ltl_reward"] = reward
        info["is_ltl_satisfied"] = self.q == 3
        info["is_ltl_dead"] = False

        return self.flatten(obs, labels), reward, terminated, truncated, info

    def label(self, obs, info):
        labels = set()

        if bool(obs["has_key"]):
            labels.add("key")

        # Fix: only label "door" on the open transition, not on revisits.
        # The fixed env sets event="door_open" only when transitioning closed→open,
        # so obs["door_open"] alone would fire every step once open.
        # We use obs["door_open"] for the persistent "agent has opened the door"
        # state (needed to correctly stay in q=2 after reaching it), but the
        # DFA transition to q=2 is gated on the event so it only fires once.
        if bool(obs["door_open"]):
            labels.add("door")

        if info.get("event", None) == "success_goal":
            labels.add("goal")

        if info.get("event", None) == "cliff":
            labels.add("unsafe")

        return labels

    def label_vector(self, labels):
        vec = np.zeros(self.n_ap, dtype=np.float32)
        for i, ap in enumerate(self.AP):
            if ap in labels:
                vec[i] = 1.0
        return vec

    def dfa_transition(self, q, labels):
        bonus = 0.0

        if "unsafe" in labels:
            bonus += self.cliff_penalty
            return q, bonus

        if q == 0:
            if "key" in labels:
                return 1, bonus + self.progress_bonus
            return 0, 0.0

        if q == 1:
            if "door" in labels:
                return 2, bonus + self.progress_bonus
            return 1, 0.0

        if q == 2:
            if "goal" in labels:
                return 3, bonus + self.progress_bonus
            return 2, 0.0

        if q == 3:
            return 3, 0.0

        return q, bonus

    def _norm_pos(self, pos):
        H, W = self.env.H, self.env.W
        return np.array(
            [pos[0] / max(H - 1, 1), pos[1] / max(W - 1, 1)],
            dtype=np.float32,
        )

    def flatten(self, obs, labels):
        pos = self._norm_pos(obs["pos"])
        flags = np.array([obs["has_key"], obs["door_open"]], dtype=np.float32)
        treasures = obs["treasures"].astype(np.float32)
        label_vec = self.label_vector(labels)

        q_onehot = np.zeros(self.n_q, dtype=np.float32)
        q_onehot[self.q] = 1.0

        return np.concatenate([pos, flags, treasures, label_vec, q_onehot])


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item(), value.item()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    values = np.append(values, 0.0)

    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = last_adv = delta + gamma * lam * nonterminal * last_adv

    returns = adv + values[:-1]
    return adv, returns


def make_env(seed=0, progress_bonus=0.5, cliff_penalty=0.0):
    # terminate_on_cliff removed — no longer supported by the fixed env
    base_env = SafeTreasureDoorKeyGrid(seed=seed)
    return LabelledLTLSafeTreasureWrapper(
        base_env,
        progress_bonus=progress_bonus,
        cliff_penalty=cliff_penalty,
    )


def evaluate(env, model, episodes=30):
    env_returns, ltl_returns, costs = [], [], []
    successes, ltls = [], []
    key_rates, door_rates, goal_rates = [], [], []
    unsafe_rates, treasure_counts = [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_env_ret = 0.0
        ep_ltl_ret = 0.0
        ep_cost = 0.0
        success = 0.0
        ltl = 0.0
        saw_key = 0.0
        saw_door = 0.0
        saw_goal = 0.0
        saw_unsafe = 0.0
        max_treasures = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)

            ep_env_ret += info["env_reward"]
            ep_ltl_ret += reward
            ep_cost += info["cost"]

            labels = info.get("labels", set())
            saw_key = max(saw_key, float("key" in labels))
            saw_door = max(saw_door, float("door" in labels))
            saw_goal = max(saw_goal, float(info.get("event", None) in ["goal_early", "success_goal"]),)
            saw_unsafe = max(saw_unsafe, float("unsafe" in labels))

            success = max(success, float(info.get("is_success", False)))
            ltl = max(ltl, float(info.get("is_ltl_satisfied", False)))

            # Treasure slice: pos(2) + flags(2) = offset 4
            treasure_start = 4
            treasure_end = 4 + env.env.n_treasures
            max_treasures = max(
                max_treasures,
                float(np.sum(obs[treasure_start:treasure_end])),
            )

            done = terminated or truncated

        env_returns.append(ep_env_ret)
        ltl_returns.append(ep_ltl_ret)
        costs.append(ep_cost)
        successes.append(success)
        ltls.append(ltl)
        key_rates.append(saw_key)
        door_rates.append(saw_door)
        goal_rates.append(saw_goal)
        unsafe_rates.append(saw_unsafe)
        treasure_counts.append(max_treasures)

    return {
        # "return" key aligns with runner.py's logs["eval_return"] mapping
        "return": float(np.mean(env_returns)),
        "env_return": float(np.mean(env_returns)),
        "ltl_return": float(np.mean(ltl_returns)),
        "cost": float(np.mean(costs)),
        "success": float(np.mean(successes)),
        "ltl": float(np.mean(ltls)),
        "key_rate": float(np.mean(key_rates)),
        "door_rate": float(np.mean(door_rates)),
        "goal_rate": float(np.mean(goal_rates)),
        "unsafe_rate": float(np.mean(unsafe_rates)),
        "treasure_count": float(np.mean(treasure_counts)),
    }


def train_labelled_ltl_ppo(
    total_steps=300_000,
    rollout_steps=1024,
    update_epochs=8,
    minibatch_size=256,
    gamma=0.99,
    lam=0.95,
    clip_coef=0.20,
    vf_coef=0.50,
    ent_coef=0.04,
    lr=3e-4,
    progress_bonus=0.5,
    cliff_penalty=0.0,
    terminate_on_cliff=False,   # kept for runner.py API compat; ignored by env
    eval_every_updates=5,
    eval_episodes=30,
    seed=0,
    save_model=False,
    verbose=True,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(
        seed=seed,
        progress_bonus=progress_bonus,
        cliff_penalty=cliff_penalty,
    )

    obs, _ = env.reset(seed=seed)
    obs_dim = len(obs)
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logs = {
        "steps": [],
        "eval_return": [],
        "eval_env_return": [],
        "eval_ltl_return": [],
        "eval_cost": [],
        "eval_success": [],
        "eval_ltl": [],
        "key_rate": [],
        "door_rate": [],
        "goal_rate": [],
        "unsafe_rate": [],
        "treasure_count": [],
    }

    global_step = 0
    update = 0

    while global_step < total_steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []

        obs, _ = env.reset()

        for _ in range(rollout_steps):
            action, logp, value = model.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)

            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()

            if global_step >= total_steps:
                break

        n = len(obs_buf)

        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)

        adv_arr, ret_arr = compute_gae(rew_arr, val_arr, done_arr, gamma, lam)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32)
        act_t = torch.tensor(act_arr, dtype=torch.long)
        old_logp_t = torch.tensor(logp_arr, dtype=torch.float32)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32)
        ret_t = torch.tensor(ret_arr, dtype=torch.float32)

        inds = np.arange(n)

        for _ in range(update_epochs):
            np.random.shuffle(inds)

            for start in range(0, n, minibatch_size):
                mb = inds[start:start + minibatch_size]

                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)

                new_logp = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - old_logp_t[mb])

                pg_loss_1 = -adv_t[mb] * ratio
                pg_loss_2 = -adv_t[mb] * torch.clamp(
                    ratio,
                    1.0 - clip_coef,
                    1.0 + clip_coef,
                )

                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                vf_loss = ((values - ret_t[mb]) ** 2).mean()

                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        update += 1

        if update % eval_every_updates == 0:
            eval_env = make_env(
                seed=seed + 10_000 + update,
                progress_bonus=progress_bonus,
                cliff_penalty=cliff_penalty,
            )

            eval_stats = evaluate(eval_env, model, episodes=eval_episodes)

            logs["steps"].append(global_step)
            # runner.py reads logs["eval_return"] for all algos
            logs["eval_return"].append(eval_stats["return"])
            logs["eval_env_return"].append(eval_stats["env_return"])
            logs["eval_ltl_return"].append(eval_stats["ltl_return"])
            logs["eval_cost"].append(eval_stats["cost"])
            logs["eval_success"].append(eval_stats["success"])
            logs["eval_ltl"].append(eval_stats["ltl"])
            logs["key_rate"].append(eval_stats["key_rate"])
            logs["door_rate"].append(eval_stats["door_rate"])
            logs["goal_rate"].append(eval_stats["goal_rate"])
            logs["unsafe_rate"].append(eval_stats["unsafe_rate"])
            logs["treasure_count"].append(eval_stats["treasure_count"])

            if verbose:
                print(
                    f"LTL-PPO-Rand | seed {seed:03d} | "
                    f"Update {update:04d} | Steps {global_step:08d} | "
                    f"EnvRet {eval_stats['env_return']:8.3f} | "
                    f"LTLRet {eval_stats['ltl_return']:8.3f} | "
                    f"Cost {eval_stats['cost']:7.3f} | "
                    f"Succ {eval_stats['success']:.2f} | "
                    f"LTL {eval_stats['ltl']:.2f} | "
                    f"K {eval_stats['key_rate']:.2f} | "
                    f"D {eval_stats['door_rate']:.2f} | "
                    f"G {eval_stats['goal_rate']:.2f} | "
                    f"Unsafe {eval_stats['unsafe_rate']:.2f} | "
                    f"T {eval_stats['treasure_count']:.2f}"
                )

    if save_model:
        torch.save(
            model.state_dict(),
            f"labelled_ltl_ppo_random_safetreasuregoal_seed{seed}.pt",
        )

    return model, logs

if __name__ == "__main__":
    train_labelled_ltl_ppo(
        total_steps=1_000_000,
        rollout_steps=2048,
        update_epochs=8,
        minibatch_size=256,
        progress_bonus=1.0,
        cliff_penalty=0.0,
        seed=0,
    )