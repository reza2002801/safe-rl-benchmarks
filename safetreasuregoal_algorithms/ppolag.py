import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from safetreasuregoal import SafeTreasureDoorKeyGrid


class ObsWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self.flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat_obs = self.flatten(obs)

        info["flat_treasure_count"] = float(np.sum(obs["treasures"]))

        return flat_obs, reward, terminated, truncated, info

    def flatten(self, obs):
        H, W = self.env.H, self.env.W
        pos = np.array(
            [
                obs["pos"][0] / max(H - 1, 1),
                obs["pos"][1] / max(W - 1, 1),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                pos,
                np.array([obs["has_key"], obs["door_open"]], dtype=np.float32),
                obs["treasures"].astype(np.float32),
            ]
        )


class ActorCriticLag(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden, act_dim)
        self.reward_critic = nn.Linear(hidden, 1)
        self.cost_critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = self.actor(z)
        reward_value = self.reward_critic(z).squeeze(-1)
        cost_value = self.cost_critic(z).squeeze(-1)
        return logits, reward_value, cost_value

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, reward_value, cost_value = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).item(),
            reward_value.item(),
            cost_value.item(),
        )


def compute_gae(x, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(x, dtype=np.float32)
    last_adv = 0.0
    values = np.append(values, 0.0)

    for t in reversed(range(len(x))):
        nonterminal = 1.0 - dones[t]
        delta = x[t] + gamma * values[t + 1] * nonterminal - values[t]
        adv[t] = last_adv = delta + gamma * lam * nonterminal * last_adv

    returns = adv + values[:-1]
    return adv, returns


def evaluate(env, model, episodes=30):
    returns = []
    costs = []
    successes = []
    ltls = []

    key_rates = []
    door_rates = []
    goal_rates = []
    unsafe_rates = []
    treasure_counts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_ret = 0.0
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
                logits, _, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)

            ep_ret += reward
            ep_cost += info["cost"]

            success = max(success, float(info.get("is_success", False)))
            ltl = max(ltl, float(info.get("is_ltl_satisfied", False)))

            saw_key = max(saw_key, float(info.get("has_key", False)))
            saw_door = max(saw_door, float(info.get("door_open", False)))
            saw_goal = max(saw_goal, float(info.get("is_success", False)))
            saw_unsafe = max(saw_unsafe, float(info.get("is_unsafe", False)))
            max_treasures = max(
                max_treasures,
                float(info.get("flat_treasure_count", 0.0)),
            )

            done = terminated or truncated

        returns.append(ep_ret)
        costs.append(ep_cost)
        successes.append(success)
        ltls.append(ltl)

        key_rates.append(saw_key)
        door_rates.append(saw_door)
        goal_rates.append(saw_goal)
        unsafe_rates.append(saw_unsafe)
        treasure_counts.append(max_treasures)

    return {
        "return": float(np.mean(returns)),
        "cost": float(np.mean(costs)),
        "success": float(np.mean(successes)),
        "ltl": float(np.mean(ltls)),
        "key_rate": float(np.mean(key_rates)),
        "door_rate": float(np.mean(door_rates)),
        "goal_rate": float(np.mean(goal_rates)),
        "unsafe_rate": float(np.mean(unsafe_rates)),
        "treasure_count": float(np.mean(treasure_counts)),
    }


def train_ppolag(
    total_steps=100_000,
    rollout_steps=1024,
    update_epochs=8,
    minibatch_size=256,
    gamma=0.99,
    cost_gamma=0.99,
    lam=0.95,
    clip_coef=0.20,
    vf_coef=0.50,
    cost_vf_coef=0.50,
    ent_coef=0.04,
    lr=3e-4,
    lag_lr=0.05,
    cost_limit=1.0,
    eval_every_updates=5,
    eval_episodes=30,
    seed=0,
    save_model=False,
    verbose=True,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env = SafeTreasureDoorKeyGrid(seed=seed)
    env = ObsWrapper(base_env)

    obs, _ = env.reset(seed=seed)
    obs_dim = len(obs)
    act_dim = env.action_space.n

    model = ActorCriticLag(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lagrangian = 0.0

    logs = {
        "steps": [],
        "eval_return": [],
        "eval_cost": [],
        "eval_success": [],
        "eval_ltl": [],
        "key_rate": [],
        "door_rate": [],
        "goal_rate": [],
        "unsafe_rate": [],
        "treasure_count": [],
        "lagrangian": [],
    }

    global_step = 0
    update = 0

    while global_step < total_steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, cost_buf, done_buf = [], [], []
        r_val_buf, c_val_buf = [], []

        obs, _ = env.reset()

        ep_costs = []
        current_ep_cost = 0.0

        for _ in range(rollout_steps):
            action, logp, r_value, c_value = model.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = float(info["cost"])

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(reward)
            cost_buf.append(cost)
            done_buf.append(done)
            r_val_buf.append(r_value)
            c_val_buf.append(c_value)

            current_ep_cost += cost
            obs = next_obs
            global_step += 1

            if done:
                ep_costs.append(current_ep_cost)
                current_ep_cost = 0.0
                obs, _ = env.reset()

            if global_step >= total_steps:
                break

        n = len(obs_buf)

        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        cost_arr = np.array(cost_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)
        r_val_arr = np.array(r_val_buf, dtype=np.float32)
        c_val_arr = np.array(c_val_buf, dtype=np.float32)

        r_adv, r_ret = compute_gae(rew_arr, r_val_arr, done_arr, gamma, lam)
        c_adv, c_ret = compute_gae(cost_arr, c_val_arr, done_arr, cost_gamma, lam)

        r_adv = (r_adv - r_adv.mean()) / (r_adv.std() + 1e-8)
        c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

        lag_adv = (r_adv - lagrangian * c_adv) / (1.0 + lagrangian)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32)
        act_t = torch.tensor(act_arr, dtype=torch.long)
        old_logp_t = torch.tensor(logp_arr, dtype=torch.float32)
        adv_t = torch.tensor(lag_adv, dtype=torch.float32)
        r_ret_t = torch.tensor(r_ret, dtype=torch.float32)
        c_ret_t = torch.tensor(c_ret, dtype=torch.float32)

        inds = np.arange(n)

        for _ in range(update_epochs):
            np.random.shuffle(inds)

            for start in range(0, n, minibatch_size):
                mb = inds[start:start + minibatch_size]

                logits, r_values, c_values = model(obs_t[mb])
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
                r_vf_loss = ((r_values - r_ret_t[mb]) ** 2).mean()
                c_vf_loss = ((c_values - c_ret_t[mb]) ** 2).mean()

                loss = (
                    pg_loss
                    + vf_coef * r_vf_loss
                    + cost_vf_coef * c_vf_loss
                    - ent_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if len(ep_costs) > 0:
            avg_ep_cost = float(np.mean(ep_costs))
        else:
            avg_ep_cost = float(np.sum(cost_arr))

        lagrangian = max(0.0, lagrangian + lag_lr * (avg_ep_cost - cost_limit))

        update += 1

        if update % eval_every_updates == 0:
            eval_env = ObsWrapper(
                SafeTreasureDoorKeyGrid(seed=seed + 10_000 + update)
            )
            eval_stats = evaluate(eval_env, model, episodes=eval_episodes)

            logs["steps"].append(global_step)
            logs["eval_return"].append(eval_stats["return"])
            logs["eval_cost"].append(eval_stats["cost"])
            logs["eval_success"].append(eval_stats["success"])
            logs["eval_ltl"].append(eval_stats["ltl"])
            logs["key_rate"].append(eval_stats["key_rate"])
            logs["door_rate"].append(eval_stats["door_rate"])
            logs["goal_rate"].append(eval_stats["goal_rate"])
            logs["unsafe_rate"].append(eval_stats["unsafe_rate"])
            logs["treasure_count"].append(eval_stats["treasure_count"])
            logs["lagrangian"].append(float(lagrangian))

            if verbose:
                print(
                    f"PPO-Lag  | seed {seed:03d} | "
                    f"Update {update:04d} | Steps {global_step:08d} | "
                    f"EvalReturn {eval_stats['return']:8.3f} | "
                    f"EvalCost {eval_stats['cost']:7.3f} | "
                    f"EvalSucc {eval_stats['success']:.2f} | "
                    f"EvalLTL {eval_stats['ltl']:.2f} | "
                    f"K {eval_stats['key_rate']:.2f} | "
                    f"D {eval_stats['door_rate']:.2f} | "
                    f"G {eval_stats['goal_rate']:.2f} | "
                    f"Unsafe {eval_stats['unsafe_rate']:.2f} | "
                    f"T {eval_stats['treasure_count']:.2f} | "
                    f"Lag {lagrangian:7.3f}"
                )

    if save_model:
        torch.save(model.state_dict(), f"ppolag_safetreasuregoal_seed{seed}.pt")

    return model, logs


if __name__ == "__main__":
    train_ppolag(
        total_steps=1_000_000,
        rollout_steps=2048,
        update_epochs=8,
        minibatch_size=256,
        cost_limit=1.0,
        lag_lr=0.01,
        seed=0,
    )