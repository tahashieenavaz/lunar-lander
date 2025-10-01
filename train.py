import torch
import warnings
import random
from collections import deque, namedtuple
from utils import (
    get_device,
    state_to_tensor,
    get_environment,
    we_can_update_model,
    soft_update,
    save_rewards,
)
from classes import Settings, QNetwork

warnings.filterwarnings("ignore")


q = QNetwork().to(get_device())
target = QNetwork().to(get_device())
target.load_state_dict(q.state_dict())

env = get_environment()
experiences = deque(maxlen=100_000)
experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)

optimizer = torch.optim.AdamW(q.parameters(), lr=Settings.lr, amsgrad=True)
criterion = torch.nn.SmoothL1Loss()
episode_rewards = []

for episode in range(Settings.episodes):
    state, _ = env.reset()
    state = state_to_tensor(state)
    episode_reward = 0.0
    done = False
    t = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q(state).argmax(dim=1).item()

        next_state, reward, truncated, terminated, _ = env.step(action)
        done = terminated or truncated
        next_state = state_to_tensor(next_state)
        experiences.append(
            experience(
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                done=done,
            )
        )
        episode_reward += reward
        state = next_state

        if we_can_update_model(experiences):
            batch = random.sample(experiences, Settings.batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states).to(get_device())
            next_states = torch.cat(next_states).to(get_device())
            actions = torch.tensor(actions, dtype=torch.long).to(get_device())
            rewards = torch.tensor(rewards, dtype=torch.float32).to(get_device())
            dones = torch.tensor(dones, dtype=torch.float32).to(get_device())

            q_values = q(states)
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = q(next_states).argmax(dim=1, keepdim=True)
                next_q_values = target(next_states).gather(1, next_actions).squeeze(1)

                y = rewards + Settings.gamma * next_q_values * (1 - dones)

            optimizer.zero_grad()
            loss = criterion(q_sa, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()
        t += 1

    soft_update(q, target)
    epsilon = max(Settings.epsilon_min, epsilon * Settings.epsilon_decay)
    episode_rewards.append((episode, episode_reward))
    print(f"Episode: {episode} Epsilon: {epsilon} Reward: {episode_reward}", flush=True)

env.close()
save_rewards(episode_rewards)
