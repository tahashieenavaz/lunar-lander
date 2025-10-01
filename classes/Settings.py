class Settings:
    gamma = 0.95
    episodes = 1000
    batch_size = 64
    epsilon = 0.9
    epsilon_min = 0.01
    epsilon_decay = (epsilon_min / epsilon) ** (1 / episodes)
    lr = 1e-3
    tau = 0.01
