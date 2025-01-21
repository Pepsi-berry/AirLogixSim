import functools
from copy import copy
from enum import Enum

import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, MultiBinary, Box, Tuple
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID, ObsType


class UAVState(Enum):
    IDLE = 0
    DELIVERING = 1
    RETURNING = 2


class CustomerState(Enum):
    WAITING = 0
    TRUCK_DELIVERED = 1
    UAV_DELIVERED = 2


class DeliveryEnv(ParallelEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "delivery_env_v1",
    }

    def __init__(self, config: dict):
        """initialize the environment.

        Attributes of config:
            - uav_num: the number of uavs in one group
            - truck_num: the number of trucks in one group
            - group_num: the number of groups
            - uav_velocity: the velocity of uavs
            - truck_velocity: the velocity of trucks
            - uav_capacity: the capacity of uavs
            - uav_power: the power of uavs
            - max_step: the maximum number of steps
            - num_customer: the number of customers
            - render_mode: the mode of rendering
            - space_width: half of the width of the space (2 * space_width + 1 = width of the space)
            - space_height: half of the height of the space (2 * space_height + 1 = height of the space)

        :param config: a dictionary containing the configuration of the environment
        """
        # random number generator
        self.RNG, _ = seeding.np_random()

        # hyperparameters
        self.uav_num = config.get("uav_num", 2)
        self.truck_num = config.get("truck_num", 1)
        self.group_num = config.get("group_num", 1)
        self.uav_velocity = config.get("uav_velocity", [12, 29])
        self.truck_velocity = config.get("truck_velocity", 7)
        self.uav_capacity = config.get("uav_capacity", 10)
        self.uav_power = config.get("uav_power", 100)
        self.max_step = config.get("max_step", 10_000)
        self.num_customer = config.get("num_customer", 20)
        self.render_mode = config.get("render_mode", "human")
        self.space_width = config.get("space_width", 100)
        self.space_height = config.get("space_height", 100)

        self.means = config.get("means", [0, 0])
        self.std_dev = config.get("std_devs", [1, 1])

        # agent definition
        self.possible_agents = ([f"uav_{i}_{j}" for j in range(self.uav_num) for i in range(self.group_num)]
                                + [f"truck_{i}_{j}" for j in range(self.truck_num) for i in range(self.group_num)])
        self.agent_name_mapping = dict(
            zip(self.possible_agents, range(len(self.possible_agents)))
        )

        # variables can be reset
        self.time_step = 0
        self.nodes_location = None
        self.nodes_weight = None
        self.warehouse = (0, 0)
        self.cur_uav_capacity = {
            agent: self.uav_capacity for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_power = {
            agent: self.uav_power for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_travel_distance = {
            agent: 0 for agent in self.possible_agents if agent.startswith("uav")
        }
        self.agent_coordinates = {
            agent: self.warehouse for agent in self.possible_agents
        }

        # define the observation space and action space for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            obs = {
                "nodes_mask": MultiBinary(self.num_customer),
                "nodes": Tuple([
                    Box(low=np.array([-self.space_width, -self.space_height]),
                        high=np.array([self.space_width, self.space_height]),
                        dtype=np.int16)
                    ] * self.num_customer)
            }
            if agent.startswith("truck"):
                obs["uav_status"] = MultiDiscrete(np.array([len(UAVState)] * self.uav_num))
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Discrete(self.num_customer + 1)
            else:
                obs["truck"] = Box(low=np.array([-self.space_width, -self.space_height]), high=np.array([self.space_width, self.space_height]), dtype=np.int16)
                obs["coordinate"] = Box(low=np.array([-self.space_width, -self.space_height]), high=np.array([self.space_width, self.space_height]), dtype=np.int16)
                obs["power"] = Box(0, high=self.uav_power, dtype=np.float32)
                obs["capacity"] = Box(0, high=self.uav_capacity, dtype=np.float32)
                obs["travel_distance"] = Box(0, high=self.space_width * self.space_height, dtype=np.float32),
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ):
        """reset the environment

        :param seed: the seed for the environment
        :param options: the options for the environment
        :return: a tuple containing the observations and info
        """
        if seed is not None:
            self.RNG, _ = seeding.np_random(seed)

        # reset the variables
        self.agents = copy(self.possible_agents)
        self.time_step = 0
        self.nodes_location = self._generate_nodes()
        self.nodes_weight = self._generate_nodes_weight()
        self.warehouse = (0, 0)
        self.cur_uav_capacity = {
            agent: self.uav_capacity for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_power = {
            agent: self.uav_power for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_travel_distance = {
            agent: 0 for agent in self.possible_agents if agent.startswith("uav")
        }
        self.agent_coordinates = {
            agent: self.warehouse for agent in self.possible_agents
        }

        observations = {}
        for agent in self.possible_agents:
            obs = {
                "nodes_mask": np.zeros(self.num_customer),
                "nodes": self.nodes_location
            }
            if agent.startswith("truck"):
                obs["uav_status"] = np.zeros(self.uav_num)
                observations[agent] = obs
            else:
                obs["truck"] = self.warehouse
                obs["coordinate"] = self.warehouse
                obs["power"] = self.uav_power
                obs["capacity"] = self.uav_capacity
                obs["travel_distance"] = 0
                observations[agent] = obs

        # return the initial observations
        return observations, {}

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def _generate_nodes(self):
        """generate the nodes in the space
        :return: the coordinates of the nodes
        """
        x_coordinates = np.clip(self.RNG.normal(self.means[0], self.std_dev[0], self.num_customer), -self.space_width, self.space_width)
        y_coordinates = np.clip(self.RNG.normal(self.means[1], self.std_dev[1], self.num_customer), -self.space_height, self.space_height)
        return np.stack((x_coordinates, y_coordinates), axis=1)

    def _generate_nodes_weight(self):
        """generate the weights of the nodes
        :return: the weights of the nodes
        """
        return self.RNG.integers(1, self.uav_capacity // 2, self.num_customer)
