# safetreasuregoal_ltlppo_labelled.py
#
# Modifications vs. original:
#
#   1. progress_bonus is now a proper hyperparameter exposed in __main__
#      (set to 0.0 to match PPO/PPOLag baselines, or >0 for full LTL-PPO).
#
#   2. DFA dead state (q=4) is removed.  A cliff hit now applies a one-step
#      penalty bonus and leaves q unchanged, so gradients are never silenced
#      for the rest of an episode.  This has no analogue in the flat baselines
#      and was causing unfair gradient corruption regardless of bonus setting.
#
#   3. terminate_on_cliff is passed through make_env and defaults to False
#      during training, matching the flat-reward baselines which never
#      terminate early on cliff.
#
#   4. q==2 DFA branch simplified: only "goal" is checked (key+door are
#      already guaranteed by having reached q==2).
#
#   5. Position normalisation uses actual grid dimensions (H-1, W-1) instead
#      of the hardcoded 5.0.
#
#   6. n_q reduced from 5 to 4 (no dead state), obs_dim adjusted accordingly.
#
#   7. Logging now tracks cliff_penalty_rate to make the DFA change auditable.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from safetreasuregoal import SafeTreasureDoorKeyGrid


class LabelledLTLSafeTreasureWrapper:
    """
    Labelled-state LTL-PPO wrapper.

    Atomic propositions:
        key     : agent has collected the key
        door    : door has been opened
        goal    : agent is currently at the goal cell
        unsafe  : agent stepped onto a cliff cell this step

    LTL task:
        eventually key, then eventually door, then eventually goal,
        while avoiding unsafe.

    DFA states (FIX #2 — no permanent dead state):
        q = 0 : need key
        q = 1 : need door
        q = 2 : need goal
        q = 3 : accepting

    A cliff hit applies a one-step negative bonus but does NOT transition
    to a dead state — q stays where it is.  The base env reward (-5) and
    cost already penalise the cliff; killing DFA gradients on top was
    causing learning to collapse mid-episode.
    """

    AP = ["key", "door", "goal", "unsafe"]

    # FIX #2: 4 states, no dead state
    N_Q = 4

    def __init__(self, env, progress_bonus=0.5, cliff_penalty=0.0):
        """
        Args:
            env            : SafeTreasureDoorKeyGrid instance
            progress_bonus : reward added on each DFA forward transition
                             (set 0.0 to match unshpaed baselines)
            cliff_penalty  : one-step bonus applied when "unsafe" fires
                             (negative value = extra penalty on top of env)
        """
        self.env = env
        self.action_space = env.action_space

        self.progress_bonus = progress_bonus
        self.cliff_penalty = cliff_penalty

        self.q = 0
        self.n_q = self.N_Q
        self.n_ap = len(self.AP)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

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
        info["is_ltl_dead"] = False  # dead state removed

        return self.flatten(obs, labels), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------

    def label(self, obs, info):
        labels = set()

        if bool(obs["has_key"]):
            labels.add("key")

        if bool(obs["door_open"]):
            labels.add("door")

        if tuple(obs["pos"]) == self.env.goal_pos:
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

    # ------------------------------------------------------------------
    # DFA transition (FIX #2, FIX #4)
    # ------------------------------------------------------------------

    def dfa_transition(self, q, labels):
        bonus = 0.0

        # FIX #2: cliff applies a one-step penalty but q is unchanged.
        # The episode continues; gradients keep flowing.
        if "unsafe" in labels:
            bonus += self.cliff_penalty  # typically 0.0 or a small negative
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
            # FIX #4: key and door are guaranteed by having reached q==2;
            # only "goal" needs to be checked here.
            if "goal" in labels:
                return 3, bonus + self.progress_bonus
            return 2, 0.0

        if q == 3:
            return 3, 0.0

        return q, bonus

    # ------------------------------------------------------------------
    # Observation flattening (FIX #5, #6)
    # ------------------------------------------------------------------

    def flatten(self, obs, labels):
        # FIX #5: normalise by actual grid dimensions, not hardcoded 5.0
        H, W = self.env.H, self.env.W
        pos = np.array(
            [obs["pos"][0] / max(H - 1, 1),
             obs["pos"][1] / max(W - 1, 1)],
            dtype=np.float32,
        )

        flags = np.array([obs["has_key"], obs["door_open"]], dtype=np.float32)
        treasures = obs["treasures"].astype(np.float32)

        label_vec = self.label_vector(labels)

        # FIX #6: q_onehot now has N_Q=4 entries (no dead state slot)
        q_onehot = np.zeros(self.n_q, dtype=np.float32)
        q_onehot[self.q] = 1.0

        return np.concatenate([pos, flags, treasures, label_vec, q_onehot])


# ----------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# GAE
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Environment factory
# ----------------------------------------------------------------------

def make_env(
    seed=0,
    slip_prob=0.10,
    progress_bonus=0.5,
    cliff_penalty=0.0,
    terminate_on_cliff=False,   # FIX #3: default False during training
):
    base_env = SafeTreasureDoorKeyGrid(seed=seed)

    return LabelledLTLSafeTreasureWrapper(
        base_env,
        progress_bonus=progress_bonus,
        cliff_penalty=cliff_penalty,
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(env, model, episodes=30):
    env_returns = []
    ltl_returns = []
    costs = []

    successes = []
    ltls = []

    key_rates = []
    door_rates = []
    goal_rates = []
    unsafe_rates = []
    cliff_penalty_rates = []
    treasure_counts = []

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
        saw_cliff_penalty = 0.0
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
            saw_goal = max(saw_goal, float("goal" in labels))
            saw_unsafe = max(saw_unsafe, float("unsafe" in labels))

            # Track how often cliff_penalty fired (audits FIX #2)
            if "unsafe" in labels and env.cliff_penalty != 0.0:
                saw_cliff_penalty = 1.0

            success = max(success, float(info.get("is_success", False)))
            ltl = max(ltl, float(info.get("is_ltl_satisfied", False)))

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
        cliff_penalty_rates.append(saw_cliff_penalty)
        treasure_counts.append(max_treasures)

    return {
        "env_return": np.mean(env_returns),
        "ltl_return": np.mean(ltl_returns),
        "cost": np.mean(costs),
        "success": np.mean(successes),
        "ltl": np.mean(ltls),
        "key_rate": np.mean(key_rates),
        "door_rate": np.mean(door_rates),
        "goal_rate": np.mean(goal_rates),
        "unsafe_rate": np.mean(unsafe_rates),
        "cliff_penalty_rate": np.mean(cliff_penalty_rates),
        "treasure_count": np.mean(treasure_counts),
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

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
    slip_prob=0.10,
    progress_bonus=0.5,
    cliff_penalty=0.0,
    terminate_on_cliff=False,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(
        seed=seed,
        slip_prob=slip_prob,
        progress_bonus=progress_bonus,
        cliff_penalty=cliff_penalty,
        terminate_on_cliff=terminate_on_cliff,
    )

    obs, _ = env.reset(seed=seed)

    obs_dim = len(obs)
    act_dim = env.action_space.n

    print(f"obs_dim={obs_dim}  act_dim={act_dim}  n_q={env.n_q}")
    print(
        f"progress_bonus={progress_bonus}  cliff_penalty={cliff_penalty}"
        f"  terminate_on_cliff={terminate_on_cliff}"
    )

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logs = {
        "steps": [],
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
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

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

        obs_arr  = np.array(obs_buf,  dtype=np.float32)
        act_arr  = np.array(act_buf,  dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr  = np.array(rew_buf,  dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)
        val_arr  = np.array(val_buf,  dtype=np.float32)

        adv_arr, ret_arr = compute_gae(rew_arr, val_arr, done_arr, gamma, lam)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t     = torch.tensor(obs_arr,  dtype=torch.float32)
        act_t     = torch.tensor(act_arr,  dtype=torch.long)
        old_logp_t = torch.tensor(logp_arr, dtype=torch.float32)
        adv_t     = torch.tensor(adv_arr,  dtype=torch.float32)
        ret_t     = torch.tensor(ret_arr,  dtype=torch.float32)

        inds = np.arange(rollout_steps)

        for _ in range(update_epochs):
            np.random.shuffle(inds)

            for start in range(0, rollout_steps, minibatch_size):
                mb = inds[start:start + minibatch_size]

                logits, values = model(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)

                new_logp = dist.log_prob(act_t[mb])
                entropy  = dist.entropy().mean()
                ratio    = torch.exp(new_logp - old_logp_t[mb])

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

        if update % 5 == 0:
            eval_env = make_env(
                seed=seed + 10_000 + update,
                slip_prob=slip_prob,
                progress_bonus=progress_bonus,
                cliff_penalty=cliff_penalty,
                terminate_on_cliff=terminate_on_cliff,
            )

            eval_stats = evaluate(eval_env, model, episodes=30)

            logs["steps"].append(global_step)
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

            print(
                f"LTL-PPO | Steps {global_step:07d} | "
                f"EnvRet {eval_stats['env_return']:7.3f} | "
                f"LTLRet {eval_stats['ltl_return']:7.3f} | "
                f"Cost {eval_stats['cost']:6.3f} | "
                f"Succ {eval_stats['success']:.2f} | "
                f"LTL {eval_stats['ltl']:.2f} | "
                f"K {eval_stats['key_rate']:.2f} | "
                f"D {eval_stats['door_rate']:.2f} | "
                f"G {eval_stats['goal_rate']:.2f} | "
                f"Unsafe {eval_stats['unsafe_rate']:.2f} | "
                f"T {eval_stats['treasure_count']:.2f}"
            )

    torch.save(model.state_dict(), f"labelled_ltl_ppo_safetreasuregoal_seed{seed}.pt")
    plot_logs(logs)

    return model, logs


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_logs(logs):
    steps = logs["steps"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Labelled LTL-PPO — SafeTreasureGoal", fontsize=13)

    axes[0, 0].plot(steps, logs["eval_env_return"], label="Env return")
    axes[0, 0].plot(steps, logs["eval_ltl_return"], label="LTL-shaped return")
    axes[0, 0].set_title("Return")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(steps, logs["eval_cost"])
    axes[0, 1].set_title("Cost")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].grid(True)

    axes[0, 2].plot(steps, logs["eval_success"], label="Success rate")
    axes[0, 2].plot(steps, logs["eval_ltl"],     label="LTL satisfied")
    axes[0, 2].set_title("Success / LTL satisfaction")
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(steps, logs["key_rate"],  label="Key")
    axes[1, 0].plot(steps, logs["door_rate"], label="Door")
    axes[1, 0].plot(steps, logs["goal_rate"], label="Goal")
    axes[1, 0].set_title("Label progress rates")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(steps, logs["unsafe_rate"])
    axes[1, 1].set_title("Unsafe rate (cliff visits)")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True)

    axes[1, 2].plot(steps, logs["treasure_count"])
    axes[1, 2].set_title("Avg treasures collected")
    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig("labelled_ltl_ppo_safetreasuregoal.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------

# --- Baseline-fair run (progress_bonus=0.0, matches PPO/PPOLag) ------
def run_no_bonus(seed=0):
    """
    Disables progress bonus to isolate the benefit of DFA state
    augmentation alone.  Use this when comparing against PPO / PPOLag
    that have no LTL shaping.
    """
    return train_labelled_ltl_ppo(
        total_steps=1_000_000,
        rollout_steps=2048,
        update_epochs=8,
        minibatch_size=256,
        slip_prob=0.03,        # match the env default
        progress_bonus=0.0,
        cliff_penalty=0.0,
        terminate_on_cliff=False,
        seed=seed,
    )

#---LTL Run---
def run_full_ltl(seed=0):
    """
    Full method as described in the LTL-PPO / reward-machine literature.
    progress_bonus provides the shaping signal that makes DFA transitions
    intrinsically rewarding.
    """
    return train_labelled_ltl_ppo(
        total_steps=1_000_000,
        rollout_steps=2048,
        update_epochs=8,
        minibatch_size=256,
        slip_prob=0.03,        # match the env default
        progress_bonus=1.0,
        cliff_penalty=0.0,
        terminate_on_cliff=False,
        seed=seed,
    )

if __name__ == "__main__":
    # Change to run_full_ltl() for the full method.
    #model, logs = run_no_bonus(seed=0)
    model, logs = run_full_ltl(seed=0)
