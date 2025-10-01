import gymnasium as gym
import pickle
import torch
import os
from classes import Settings


def get_device():
    """
    The function `get_device()` returns the appropriate device based on the availability of CUDA, MPS,
    or defaults to CPU.
    :return: The `get_device` function returns the appropriate device based on availability. It returns
    "cuda" if CUDA is available, "mps" if the MPS backend is available, and "cpu" if neither CUDA nor
    MPS is available.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def save_variable(variable, filename: str):
    """
    The function `save_variable` saves a Python variable to a file using pickle serialization.

    :param variable: The `variable` parameter in the `save_variable` function is the data that you want
    to save to a file. This data can be of any type - it could be a number, string, list, dictionary,
    object, etc. The function will use the `pickle` module to serialize this
    :param filename: The `filename` parameter is a string that represents the name of the file where the
    variable will be saved using the `pickle` module
    :type filename: str
    """
    pickle.dump(variable, open(filename, "wb"))


def state_to_tensor(state):
    """
    The function `state_to_tensor` converts a state into a PyTorch tensor with float32 data type and
    adds a batch dimension.

    :param state: The `state` parameter is typically a representation of the current state of an
    environment in a reinforcement learning setting. It could be a vector, array, or any other data
    structure that contains information about the environment's state, such as the positions of objects,
    velocities, or any other relevant information needed for
    :return: The function `state_to_tensor` returns a PyTorch tensor created from the input `state`,
    with a data type of `torch.float32`, unsqueezed along the first dimension, and moved to the device
    specified by `get_device()`.
    """
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(get_device())


def we_can_update_model(experiences):
    """
    The function `we_can_update_model` returns `True` if the number of experiences is greater than twice
    the batch size specified in the `Settings`.

    :param experiences: The `experiences` parameter likely refers to a collection of data or instances
    that are used to train a machine learning model. This function `we_can_update_model` checks if the
    number of experiences is greater than twice the `batch_size` specified in the `Settings` module. If
    this condition is
    :return: The function `we_can_update_model` returns a boolean value indicating whether the number of
    experiences is greater than twice the batch size specified in the `Settings` module.
    """
    return len(experiences) > 2 * Settings.batch_size


def get_environment():
    """
    The function `get_environment` creates a gym environment for the LunarLander-v3 game with video
    recording capabilities.
    :return: The function `get_environment()` returns an environment for the LunarLander-v3 gym
    environment with video recording enabled. The environment is set to render in RGB array format and
    record videos of the episodes with the name prefix "lunarlanding".
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, "videos", name_prefix="lunarlanding", episode_trigger=lambda id: True
    )
    return env


def soft_update(q, target) -> None:
    target_state_dict = target.state_dict()
    q_state_dict = q.state_dict()
    for key in q_state_dict:
        target_state_dict[key] = q_state_dict[key] * Settings.tau + target_state_dict[
            key
        ] * (1 - Settings.tau)
    target.load_state_dict(target_state_dict)


def save_rewards(rewards):
    os.makedirs("results", exists_ok=True)
    save_variable(rewards, "./results/rewards.pkl")
    rewards.sort(key=lambda x: x[1], reverse=True)
    save_variable(rewards, "./results/rewards-sorted.pkl")
