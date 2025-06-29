from .manager.traffic_manager import TrafficManager
from .manager.task_manager import TaskManager
from .entities.vehicle import Vehicle
from .entities.uav import UAV
from .airlogixsim_visual import AirLogixSimEnvVisualizer
import traci
import numpy as np
import time
from .utils.tk_utils import parse_location_info
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Box, Tuple
from gymnasium.utils import seeding
from copy import copy
from .utils.enums import UAVState, VehicleState, PackageState,UAVActionRet
import functools
import re
from sklearn.cluster import KMeans

class AirLogixSimEnv():
    """AirLogixSimEnv is the main class for the airlogixsim environment. It provides the simulation of urban traffic flow, package generation and assignment, delivery vehicles/UAVs mobility, and AI models for entities. It also provides the APIs for the agent to interact with the environment. The agent can be a DRL agent, a rule-based agent, or a human player.
    """

    def __init__(self, config, interactive_mode=None):
        """The constructor of the AirLogixSimEnv class. It initializes the environment with the given configuration.

        Args:
            config (dict): The configuration of the environment. Please follow standard YAML format.
            interactive_mode (str, optional): The interactive mode. 'graphic' or 'text'. Defaults to None.
        """
        self.force_quit = False

        self.vehicles = {}
        self.vehicle_ids_as_index = []  # vehicle ids as a list, used for indexing

        self.UAVs = {}
        self.UAV_ids_as_index = []  # UAV ids as a list, used for indexing

        self.config = config

        self.simulation_time = 0
        self.max_simulation_time = config['simulation']['max_simulation_time']

        self.simulation_interval = config['simulation']['simulation_interval']
        self.traffic_interval = config['simulation']['traffic_interval']

        assert self.traffic_interval >= self.simulation_interval, "The traffic interval should be greater than or equal to the simulation interval!"

        self.traci_connection = self._connectToSUMO(config['sumo'], config['traffic']['traffic_mode'] == 'SUMO')
        conv_boundary, _, _, _ = parse_location_info(config['sumo']['sumo_net'])
        conv_boundary = tuple(map(float, conv_boundary.split(',')))
        config['traffic']['x_range'] = [conv_boundary[0], conv_boundary[2]]
        config['traffic']['y_range'] = [conv_boundary[1], conv_boundary[3]]

        # Set default battery settings if not provided
        if 'UAV_battery_capacity' not in config['traffic']:
            config['traffic']['UAV_battery_capacity'] = 100
        if 'UAV_energy_consumption_rate' not in config['traffic']:
            config['traffic']['UAV_energy_consumption_rate'] = 1

        # 配置管理
        self._configManagersModels()

        # 获取无人机和卡车的数量
        self.UAV_count = config['traffic']['UAV_count']
        self.vehicle_count = config['traffic']['vehicle_count']

        vehicle_traffic_infos = self.traffic_manager.getVehicleTrafficInfos()
        print("vehicle_traffic_infos", vehicle_traffic_infos)
        UAV_traffic_infos = self.traffic_manager.getUAVTrafficInfos()
        print("UAV_traffic_infos", UAV_traffic_infos)

        # Initialize vehicles if first simulation step
        if self.simulation_time == 0:
            for vehicle_id, vehicle_traffic_info in vehicle_traffic_infos.items():
                self.vehicles[vehicle_id] = self._initVehicle(vehicle_traffic_info)
            
            ## Initialize UAVs
            for UAV_id, UAV_traffic_info in UAV_traffic_infos.items():
                battery_capacity = UAV_traffic_info.get('battery_capacity', config['traffic']['UAV_battery_capacity'])
                energy_consumption_rate = UAV_traffic_info.get('energy_consumption_rate', config['traffic']['UAV_energy_consumption_rate'])
                
                self.UAVs[UAV_id] = UAV(UAV_id, UAV_traffic_info['position'], UAV_traffic_info['speed'],
                                    UAV_traffic_info['acceleration'], UAV_traffic_info['angle'],
                                    UAV_traffic_info['phi'], battery_capacity, energy_consumption_rate)
                ## set the docking info
                self.UAVs[UAV_id].dock(UAV_traffic_info['docked_vehicle_id'])

        # 可视化
        self._visualizer = None
        if interactive_mode is not None:
            self.mountVisualizer(interactive_mode)

        # ---------------- reinforcement learning settings----------------

        # some of the hyperparameters
        self.group_num = config['traffic']['group_num'] # the number of groups of vehicles and UAV, currently only support 1

        # warehouse position, if not provided, set to [0, 0, 0]
        self.warehouse_position = config['traffic']['warehouse_position'] if config['traffic']['warehouse_position'] is not None else [0, 0, 0]
        self.warehouse_position_2d = [self.warehouse_position[0], self.warehouse_position[1]]
        self.cluster_number = config['clustering']['cluster_number'] if config['clustering']['cluster_number'] is not None else 5
        self.power_coefficient = config['traffic']['UAV_energy_consumption_rate'] if config['traffic']['UAV_energy_consumption_rate'] is not None else 0.01
        self.low_power_threshold = config['traffic']['low_power_threshold'] if config['traffic']['low_power_threshold'] is not None else 0.5
        # uav满电状态下的电量
        self.uav_power = config['traffic']['UAV_battery_capacity'] if config['traffic']['UAV_battery_capacity'] is not None else 100

        # config the reward function
        self.reward_dict = {
            "every_time_step": -0.1,
            "uav_charging": lambda power: 0 if power < self.low_power_threshold * self.uav_power else 0,
            "uav_deliver_node": 0,
            "mission_completed": (lambda: max(self.max_simulation_time - self.simulation_time, 0))(),
            "uav_trip_delivery_bonus": lambda uav_name: 5 * (self.delivery_per_trip[uav_name] - 1),
            "uav_round_trip": lambda distance, load: 0,
            "uav_out_of_power": -10,
            "uav_out_of_capacity": -10,
            "uav_out_of_cluster": -10,
            "uav_out_of_power_return": -10,
            "uav_closed_nodes": -5,
            "uav_current_target": 0,
            "vehicle_current_target": 0,
            "vehicle_serve_customer": 0,
            "vehicle_illegal_return": 0
        }

        # agent definition
        self.possible_agents = [] # store all the agents' ids
        # store the uavs first
        for uav_id in self.UAVs.keys():
            self.possible_agents.append(uav_id)
        # store the vehicles
        for vehicle_id in self.vehicles.keys():
            self.possible_agents.append(vehicle_id)
        
        self.agents = copy(self.possible_agents)

        # some other rl related variables, node here refers to the delivery tasks, or in other words the customers location
        self.nodes_location = self.task_manager.get_task_locations_2d()
        self.nodes_weight = self.task_manager.get_task_weights()    
        self.agent_status = None
        self.agent_target = None
        self.node_mask = np.zeros(self.task_manager.get_num_of_tasks())
        self.terminations = None
        self.truncation = None
        self.infos = None
        self.kmeans = None
        self.group_assigned = None  # group -> k-means cluster mapping
        self.delivery_per_trip = None  # count number of nodes delivered by uav in one trip to encourage multi-visit.

        # define the observation space and action space for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            obs = {
                "action_mask": Discrete(2),
                # "node_mask": MultiBinary(self.num_customer),
                "nodes": Tuple([
                                   Box(low=np.array([config['traffic']['x_range'][0], config['traffic']['y_range'][0]]),
                                       high=np.array([config['traffic']['x_range'][1], config['traffic']['y_range'][1]]),
                                       dtype=np.float32)
                               ] * self.task_manager.get_num_of_tasks()),
                "parcel": Tuple([Box(low=0, high=self.config['traffic']['UAV_capacity'], dtype=np.float16)] * self.task_manager.get_num_of_tasks()),
                "coordinate": Box(
                    low=np.array([config['traffic']['x_range'][0], config['traffic']['y_range'][0]]),
                    high=np.array([config['traffic']['x_range'][1], config['traffic']['y_range'][1]]),
                    dtype=np.float32
                )
            }
            if agent in self.vehicles: # vehicle observation
                obs["uav_status"] = MultiDiscrete(np.array([len(UAVState)] * self.UAV_count))
                obs["node_mask"] = MultiDiscrete([len(PackageState)] * self.task_manager.get_num_of_tasks())
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Discrete(self.task_manager.get_num_of_tasks() + 1) # the action means to assign a task destination to the vehicle, while the plus one is the warehouse position
            elif agent in self.UAVs: # uav observation
                obs["truck"] = Tuple([
                                         Box(
                                             low=np.array([config['traffic']['x_range'][0], config['traffic']['y_range'][0]]),
                                             high=np.array([config['traffic']['x_range'][1], config['traffic']['y_range'][1]]),
                                             dtype=np.float32
                                         )
                                     ] * self.vehicle_count)
                # the uav can choose to go to the truck or the task node
                obs["choice_mask"] = MultiDiscrete([len(UAVActionRet)] * (self.vehicle_count + self.task_manager.get_num_of_tasks()))
                obs["power"] = Box(0, high=config['traffic']['UAV_battery_capacity'], dtype=np.float32)
                obs["capacity"] = Box(0, high=config['traffic']['UAV_capacity'], dtype=np.float32)
                obs["travel_distance"] = Box(0, high=(config['traffic']['x_range'][1] - config['traffic']['x_range'][0]) * (config['traffic']['y_range'][1] - config['traffic']['y_range'][0]), dtype=np.float32)
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Discrete(self.task_manager.get_num_of_tasks() + self.vehicle_count)
        # ----------------decisions, managed by schedulers----------------
        self.vehicle_mobility_patterns = {}  # dict, key是vehicle_id, value是mobility pattern={speed}
        self.UAV_mobility_patterns = {}  # dict, key是UAV_id, value是mobility pattern={angle, phi, speed}
        self.UAV_routes={} # dict, key是UAV_id,value是route -> [{position: [x,y,z]},{to_stay_time: time}],...]

        # ----------------indicators, managed by evaluation----------------
        self.init_indicators()

    def init_indicators(self):
       pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self,
              seed: int | None = None,
              options: dict | None = {"redistribute": True},
              ):
        """Reset the environment.
        """

        # reset the variables
        self.agents = copy(self.possible_agents)
        self.simulation_time = 0
        self.infos = {"cur_time_step": self.simulation_time, "max_step": self.max_simulation_time, "num_customer": self.task_manager.get_num_of_tasks()}

        print("num_tasks", self.config['task']['num_tasks'])
        if self.nodes_location is None or options.get("redistribute", True):
            while True:
                print("redistribute")
                self.kmeans = None
                self.task_manager.initialize_tasks(self.config['task']['num_tasks'])
                print("task_manager", self.task_manager)
                self.nodes_location = self.task_manager.get_task_locations_2d()
                print("nodes_location", self.nodes_location)
                self.nodes_weight = self.task_manager.get_task_weights()
                print("nodes_weight", self.nodes_weight)
                self._k_means_cluster()
                if self._check_feasible():
                    break
        else:
            self._k_means_cluster()  # get center_nodes
        
        self.node_mask = np.array([PackageState.WAITING.value] * self.task_manager.get_num_of_tasks())
        self.agent_status = {agent: VehicleState.LANDING.value if agent in self.vehicles else UAVState.INIT.value for agent in self.possible_agents}
        self.agent_target = {agent: None for agent in self.possible_agents}
        self.delivery_per_trip = {agent: 0 for agent in self.possible_agents if agent in self.UAVs}
        self.group_assigned = {group_num: None for group_num in range(self.group_num)}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncation = False
        observations = self._get_observations()

        self.close()
        # time.sleep(3)
        self.traci_connection = self._connectToSUMO(self.config['sumo'], self.config['traffic']['traffic_mode'] == 'SUMO')
        self.traffic_manager.reset(self.traci_connection)


        self.force_quit = False

        return observations, self.infos

    def _configManagersModels(self):
        '''
        config the managers and models
        '''
        config = self.config
        # 1. Config the traffic manager
        self.traffic_manager = TrafficManager(config['traffic'], self.traci_connection, config['sumo']['sumo_net'])
        # 2. TODO: Config the task manager
        self.task_manager = TaskManager.from_config(config, self.traffic_manager)
        self.task_manager.initialize_tasks(config['task']['num_tasks']) #generate tasks with a predefined number


    def mountVisualizer(self, mode='graphic'):
        """Mount the visualizer to the environment.

        Args:
            mode (str, optional): The mode of the visualizer. 'graphic' or 'text'. Defaults to 'graphic
        """
        self._visualizer = AirLogixSimEnvVisualizer(mode=mode, config=self.config, env=self)

    def render(self):
        """Render the environment if the visualizer is mounted.
        """
        if self._visualizer is not None:
            self._visualizer.render(self)

    @property
    def airlogixsim_label(self):
        return self._sumo_label

    def close(self):
        """Close the environment. 关闭环境，关闭SUMO连接
        """
        if self.config['traffic']['traffic_mode'] == 'SUMO':
            traci.close()
        else:
            self.traffic_manager.reset()

    def _connectToSUMO(self, config, useSUMO=True):
        """Connect to the SUMO simulator with a generated label (e.g., airlogixsim_{timestamp}).

        Args:
            config (dict): The configuration of the SUMO simulator.
        """
        if useSUMO:
            self._sumo_label = "airlogixsim_" + str(time.time())
            cmd_list = ["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", config['sumo_config']]
            if config['export_tripinfo']:
                # cmd_list.append("--tripinfo-output")
                # cmd_list.append(config['tripinfo_output'])
                # cmd_list.append("--duration-log.statistics")
                # --full-output 
                cmd_list.append("--full-output")
                cmd_list.append(config['tripinfo_output'])
            print(cmd_list)
            traci.start(cmd_list,
                        port=config['sumo_port'], label=self._sumo_label)
            assert self.traffic_interval == traci.simulation.getDeltaT(), "The traffic interval is not equal to the simulation step in SUMO!"
            traci_connection = traci.getConnection(self._sumo_label)
            return traci_connection
        else:
            self._sumo_label = "non_sumo_" + str(time.time())
        return None

    def isDone(self):
        """Check whether the environment is done.

        Returns:
            bool: The done signal. True if the episode is done, False otherwise.
        """
        return self.simulation_time >= self.max_simulation_time or self.force_quit == True

    def step(self, action, training=False):
        """take a step in the environment
        For trucks, it can choose customer nodes and warehouse as its target. The action should range from [0, num_customer].
        For uavs, it can choose customer nodes and trucks as its target. The action should range from [0, num_customer + truck_num).

        :param training: whether the step is for training
        :param action: the index of nodes.
        """
        print()
        print("first step")
        print()
        rewards = {agent: 0 for agent in self.possible_agents}
        observations = self._get_observations()

        # update agent targets
        # if choose the current target, give a negative reward
        for agent, act in action.items():
            print("agent", agent)
            print("act", act)
            group_num, _ = self._get_agent_group(agent)

            agent_coordinate = self.UAVs[agent].getPosition2D() if agent in self.UAVs else self.vehicles[agent].getPosition2D()
            if not self._get_action_mask(agent):
                print("no action mask")
                continue
            elif act < self.task_manager.get_num_of_tasks():  # customer
                if self.node_mask[act] != PackageState.WAITING.value:  # the node has been assigned or delivered
                    rewards[agent] += self.reward_dict["uav_closed_nodes"]
                    print("the node has been assigned or delivered")
                    continue
                elif np.array_equal(np.array(agent_coordinate), self.nodes_location[act]):
                    rewards[agent] += self.reward_dict["uav_current_target"]
                    continue
                # self.node_mask[act] = PackageState.ASSIGNED.value
                # self.agent_target[agent] = act
                if agent in self.UAVs:  # if uav decides to deliver a new parcel, update the power and capacity
                    if all(x != UAVActionRet.FEASIBLE.value for x in observations[agent]["choice_mask"]):
                        pass  # dead
                    elif observations[agent]["choice_mask"][act] == UAVActionRet.OUT_OF_CAPACITY.value:
                        rewards[agent] += self.reward_dict["uav_out_of_capacity"]
                        continue
                    elif observations[agent]["choice_mask"][act] == UAVActionRet.OUT_OF_POWER.value:
                        rewards[agent] += self.reward_dict["uav_out_of_power"]
                        continue
                    elif observations[agent]["choice_mask"][act] == UAVActionRet.OUT_OF_CLUSTER.value:
                        rewards[agent] += self.reward_dict["uav_out_of_cluster"]
                        continue
                    elif observations[agent]["choice_mask"][act] == UAVActionRet.CANNOT_RETURN.value:
                        rewards[agent] += self.reward_dict["uav_out_of_power_return"]
                        continue
                    self.node_mask[act] = PackageState.ASSIGNED.value
                    self.agent_target[agent] = act
                    destination3d = (self.nodes_location[act][0], self.nodes_location[act][1], 0)
                    self.traffic_manager.setUAVDestination(agent, destination3d)
                    self.traffic_manager.changeUAVCapacity(agent, self.traffic_manager.getCurrentUAVCapacity(agent) - self.nodes_weight[act])
                    group_num, _ = self._get_agent_group(agent)
                    # self.truck_loaded_uav[f"truck_{group_num}_0"].discard(agent)  # unregister uav
                    self.traffic_manager.takeOffUAV(agent)
                    self.takeOffUAV(agent)
                    self.agent_status[agent] = UAVState.DELIVERING.value
                else:
                    print("set new destination for vehicle")
                    self.node_mask[act] = PackageState.ASSIGNED.value
                    self.agent_target[agent] = act
                    self.traffic_manager.setVehicleDestination(agent, self.nodes_location[act])
                    #self.traffic_manager.resumeVehicle(agent)
                    self.agent_status[agent] = VehicleState.MOVING.value
                    self.group_assigned[group_num] = self._get_k_means_cluster(self.nodes_location[act])
                    # beware the vehicle in SUMO is stopped, so we need to resume it
                    self.traffic_manager.resumeVehicle(agent)
            elif agent in self.vehicles and act == self.task_manager.get_num_of_tasks():  # return to warehouse
                if self.traffic_manager._arrive_near_destination(agent, self.warehouse_position_2d):
                    rewards[agent] += -1
                    continue
                self.agent_status[agent] = VehicleState.MOVING.value
                self.agent_target[agent] = "warehouse"
                self.traffic_manager.setVehicleDestination(agent, self.warehouse_position_2d)
                self.traffic_manager.stopVehicle(agent, 1000000)
                self.group_assigned[group_num] = None
            elif agent in self.UAVs and act < self.task_manager.get_num_of_tasks() + self.vehicle_count:
                # UAV returns to vehicle
                group_num, _ = self._get_agent_group(agent)
                vehicle_target = f"vehicle_{group_num}_{act - self.task_manager.get_num_of_tasks()}"
                
                if np.array_equal(np.array(self.UAVs[agent].getPosition2D()),
                                  self.vehicles[vehicle_target].getPosition2D()):
                    # Already at vehicle position, dock immediately
                    self.traffic_manager.dockUAV(agent, vehicle_target)
                    self.dockUAV(agent, vehicle_target)
                    self.agent_status[agent] = UAVState.IDLE.value
                    self.agent_target[agent] = None
                    rewards[agent] += self.reward_dict["vehicle_current_target"]
                    continue
                
                # Set UAV to return to vehicle
                print(f"UAV {agent} commanded to return to {vehicle_target}")
                self.agent_status[agent] = UAVState.RETURNING.value
                self.agent_target[agent] = vehicle_target
                
                # Set destination to vehicle position
                vehicle_position = self.vehicles[vehicle_target].getPosition2D()
                destination3d = (vehicle_position[0], vehicle_position[1], 0)
                print(f"Setting UAV {agent} destination to vehicle at {vehicle_position}")
                self.traffic_manager.setUAVDestination(agent, destination3d)
            else:
                raise ValueError(f"Unknown action: {act}, agent: {agent}")
        

        print("complete updating targets for all the agents")

        first_it = True
        self.simulation_time += self.simulation_interval
        self.infos["cur_time_step"] = self.simulation_time
        while_count = 0
        while all(not self._get_action_mask(agent) or self.terminations[agent] for agent in self.possible_agents):
            self.simulation_time += self.simulation_interval if not first_it else 0
            first_it = False
            self.infos["cur_time_step"] = self.simulation_time
            if self.simulation_time > self.max_simulation_time:
                self.truncation = True
                observations = self._get_observations()
                rewards = {agent: -100 for agent in self.possible_agents}
                terminations = {agent: True for agent in self.possible_agents}
                return observations, rewards, terminations, self.truncation, self.infos

            for agent in self.possible_agents:
                rewards[agent] += self.reward_dict["every_time_step"] if self.agent_target[agent] is not None else 0

            # do the traffic simulation
            self._updateTraffics()
            # move to target
            for agent, target in self.agent_target.items():
                # print the current location of the agent
                # print every 100 iteration
                # if while_count % 100 == 0:
                #     print(f"current location of the agent:{agent}")
                #     print(self.UAVs[agent].getPosition2D() if agent in self.UAVs else self.vehicles[agent].getPosition2D())
                if target is None:
                    continue
                agent_coordinate = self.UAVs[agent].getPosition2D() if agent in self.UAVs else self.vehicles[agent].getPosition2D()

                if type(target) is int:
                    target_coordinate = self.nodes_location[target]
                elif target == 'warehouse':
                    target_coordinate = self.warehouse_position_2d
                elif target.startswith("vehicle"):
                    target_coordinate = self.vehicles[target].getPosition2D()
                else:
                    raise ValueError(f"Unknown target: {target}, agent: {agent}")

                distance = np.linalg.norm(np.array(target_coordinate) - np.array(agent_coordinate))
                x_distance = target_coordinate[0] - agent_coordinate[0]
                y_distance = target_coordinate[1] - agent_coordinate[1]

                if agent in self.UAVs:
                    if self.traffic_manager.getCurrentUAVCapacity(agent) < 0 or self.traffic_manager.getCurrentUAVPower(agent) <= 0:  # dead
                        rewards[agent] += self.reward_dict["uav_out_of_power"]
                        if type(target) is int and target < self.task_manager.get_num_of_tasks():
                            self.node_mask[target] = PackageState.WAITING.value
                        self.agent_status[agent] = UAVState.IDLE.value
                        self.agent_target[agent] = None
                        group_num, _ = self._get_agent_group(agent)
                        self.traffic_manager.dockUAV(agent, f"vehicle_{group_num}_0")
                        self.dockUAV(agent, f"vehicle_{group_num}_0")
                    elif type(target) is str and target.startswith(
                            "vehicle") and distance <= 1:  # return to truck
                        print(f"UAV {agent} has returned to vehicle {target}")
                        group_num, _ = self._get_agent_group(agent)

                        rewards[agent] += self.reward_dict["uav_round_trip"](self.traffic_manager.getCurrentUAVTravelDistance(agent),
                                                                             self.traffic_manager.getCurrentUAVCapacity(agent))
                        rewards[agent] += self.reward_dict["uav_charging"](self.traffic_manager.getCurrentUAVPower(agent))
                        rewards[agent] += self.reward_dict["uav_trip_delivery_bonus"](agent)
                        self.delivery_per_trip[agent] = 0

                        self.agent_status[agent] = UAVState.IDLE.value
                        self.agent_target[agent] = None

                        self.traffic_manager.dockUAV(agent, target)
                        self.dockUAV(agent, target)
                    elif distance <= 1:
                        print("uav has reached the target")
                        # reached the target, the parcel is delivered, so we need to change the UAV's capacity
                        self.traffic_manager.changeUAVCapacity(agent, self.traffic_manager.getCurrentUAVCapacity(agent) + self.nodes_weight[target])
                        self.node_mask[target] = PackageState.DELIVERED.value
                        self.delivery_per_trip[agent] += 1
                        rewards[agent] += self.reward_dict["uav_deliver_node"]
                        
                        # UAV waits for next decision after delivery
                        self.agent_status[agent] = UAVState.LANDING.value
                        self.agent_target[agent] = None
                    else:
                        # self.agent_coordinates[agent] = (
                        #     agent_coordinate[0] + x_distance * self.uav_velocity / distance,
                        #     agent_coordinate[1] + y_distance * self.uav_velocity / distance)
                        # self.cur_uav_travel_distance[agent] += self.uav_velocity

                        # cur_uav_load = 0 if type(target) is str and target.startswith("truck") else self.nodes_weight[
                        #     target]
                        # self.cur_uav_power[agent] -= self._calc_power_consumption(cur_uav_load + self.uav_weight,
                        #                                                           self.uav_velocity)
                        pass
                else: # vehicle
                    group_num, _ = self._get_agent_group(agent)
                    if self.traffic_manager._arrive_near_destination(agent, target_coordinate):
                        self.agent_status[agent] = VehicleState.LANDING.value
                        self.agent_target[agent] = None
                        if target != 'warehouse':
                            self.node_mask[target] = PackageState.DELIVERED.value
                            for uav in self.traffic_manager.getOnBoardUAVs(agent):
                                self.terminations[uav] = False  # open for delivery
                                self.agent_status[uav] = UAVState.IDLE.value
                            rewards[agent] += self.reward_dict["vehicle_serve_customer"]
                            # stop the vehicle
                            self.traffic_manager.stopVehicle(agent, 1000000)
                        else:
                            self.agent_status[agent] = VehicleState.LANDING.value
                            if np.all(self.node_mask == PackageState.DELIVERED.value):
                                for every_agent in self.possible_agents:
                                    rewards[every_agent] += self.reward_dict["mission_completed"]
                                self.terminations[agent] = True
                            else:
                                rewards[agent] += self.reward_dict["vehicle_illegal_return"]
                            self.traffic_manager.stopVehicle(agent, 1000000)

            # Only when all clusters are delivered, all uavs return to the truck and truck is at warehouse, the truck can move.
            for agent in self.possible_agents:
                group_num, _ = self._get_agent_group(agent)
                if agent in self.UAVs and self.group_assigned[group_num] is None and all(
                        x == PackageState.DELIVERED.value for x in self.node_mask):
                    self.terminations[agent] = True
                elif agent in self.UAVs and self.group_assigned[group_num] is None:
                    self.terminations[agent] = False
                elif agent in self.UAVs:
                    # self.terminations[agent] = self._is_cluster_delivered(
                    #     self.group_assigned[group_num]) and self._get_action_mask(agent)
                    self.terminations[agent] = self._is_cluster_delivered(
                        self.group_assigned[group_num]) and self.agent_status[agent] == UAVState.IDLE.value
                else:  # truck
                    self.terminations[agent] = (self._get_action_mask(agent)
                                                and self.traffic_manager._arrive_near_destination(agent, self.warehouse_position_2d)
                                                and all(
                                self._is_cluster_delivered(cluster) for cluster in range(self.cluster_number)))

            observations = self._get_observations()
            while_count += 1
            if not training or all(self.terminations[agent] for agent in self.possible_agents):
                break
            else:
                self.render()

        sim_step_per_traffic_step = int(self.traffic_interval / self.simulation_interval)

        print("finish one schedule step")

        return observations, rewards, self.terminations, self.truncation, self.infos
    
    def _get_observations(self):
        """Get the observation for the current state.
        """
        # print("enter get observation")
        observations = {}
        # print(self.nodes_location)
        for agent in self.possible_agents:
            group_num, agent_num = self._get_agent_group(agent)

            obs = {
                "action_mask": int(self._get_action_mask(agent)),
                "nodes": self.nodes_location,
                "parcel": self.nodes_weight,
                "coordinate": self.UAVs[agent].getPosition2D() if agent in self.UAVs else self.vehicles[agent].getPosition2D()
            }
            if agent in self.vehicles: # the observation for the vehicle
                obs["uav_status"] = np.zeros(self.UAV_count)
                obs["node_mask"] = self.node_mask
                for x in range(self.UAV_count):
                    obs["uav_status"][x] = self.agent_status[f"UAV_{group_num}_{x}"]
                observations[agent] = obs
            else: # the observation for the uav
                obs["truck"] = np.zeros((self.vehicle_count, 2))
                for x in range(self.vehicle_count):
                    obs["truck"][x] = np.array(self.vehicles[f"vehicle_{group_num}_{x}"].getPosition2D())
                obs["power"] = self.traffic_manager.getCurrentUAVPower(agent)
                obs["capacity"] = self.traffic_manager.getCurrentUAVCapacity(agent)
                obs["travel_distance"] = self.traffic_manager.getCurrentUAVTravelDistance(agent)
                obs["choice_mask"] = np.zeros(self.task_manager.get_num_of_tasks() + self.vehicle_count)
                for x in range(self.task_manager.get_num_of_tasks() + self.vehicle_count):
                    if x < self.task_manager.get_num_of_tasks():
                        if self.node_mask[x] != PackageState.WAITING.value:
                            obs["choice_mask"][x] = UAVActionRet.CLOSED_NODE.value
                        elif self.group_assigned[group_num] is None or self.group_assigned[group_num] is not None and self._get_k_means_cluster(
                                self.nodes_location[x]) != self.group_assigned[group_num]:
                            obs["choice_mask"][x] = UAVActionRet.OUT_OF_CLUSTER.value
                        elif (self.traffic_manager.getCurrentUAVPower(agent) - self._calc_power_consumption(
                                (self.nodes_weight[x] + self.traffic_manager.getUAVWeight()), (self.traffic_manager.getCurrentUAVTravelDistance(agent) +
                                                                           self._calc_distance(
                                                                               self.UAVs[agent].getPosition2D(),
                                                                               self.nodes_location[x]))) <= 0):
                            obs["choice_mask"][x] = UAVActionRet.OUT_OF_POWER.value
                        elif self.traffic_manager.getCurrentUAVCapacity(agent) - self.nodes_weight[x] < 0:
                            obs["choice_mask"][x] = UAVActionRet.OUT_OF_CAPACITY.value
                        else:
                            delivery_power = self._calc_power_consumption((self.nodes_weight[x] + self.traffic_manager.getUAVWeight()),
                                                                          (self.traffic_manager.getCurrentUAVTravelDistance(agent) + self._calc_distance(
                                                                              self.UAVs[agent].getPosition2D(),
                                                                              self.nodes_location[x])))
                            return_power = self._calc_power_consumption(self.traffic_manager.getUAVWeight(), self._calc_distance(
                                self.nodes_location[x], self.vehicles[f"vehicle_{group_num}_0"].getPosition2D()))
                            if self.traffic_manager.getCurrentUAVPower(agent) - delivery_power - return_power <= 0:
                                obs["choice_mask"][x] = UAVActionRet.OUT_OF_POWER.value
                            else:
                                obs["choice_mask"][x] = UAVActionRet.FEASIBLE.value
                    else:
                        if np.array_equal(np.array(self.UAVs[agent].getPosition2D()), self.vehicles[f"vehicle_{group_num}_{x-self.task_manager.get_num_of_tasks()}"].getPosition2D()):
                            obs["choice_mask"][x] = UAVActionRet.SAME_TARGET.value
                        else:
                            return_power = self._calc_power_consumption(self.traffic_manager.getUAVWeight(), self._calc_distance(
                                self.UAVs[agent].getPosition2D(), self.vehicles[f"vehicle_{group_num}_0"].getPosition2D()))
                            if self.traffic_manager.getCurrentUAVPower(agent) - return_power <= 0:
                                obs["choice_mask"][x] = UAVActionRet.OUT_OF_POWER.value
                            else:
                                obs["choice_mask"][x] = UAVActionRet.FEASIBLE.value
                observations[agent] = obs

        return observations
    
    def _get_action_mask(self, agent):
        """Get the action mask for the current state.
        """
        """get the action mask of the agent.

        - One UAV can move whren all the trucks ae landing, and it is idle.
        - One truck can move when all the UAVs are idle, and it is landing.

        :param agent: the name of the agent
        :return: the action mask of the agent
        """
        status = self.agent_status[agent]
        group_num, agent_num = self._get_agent_group(agent)

        if agent in self.vehicles:
            if self.group_assigned[group_num] is None:
                return status == VehicleState.LANDING.value and all(
                    self.agent_status[f"UAV_{group_num}_{i}"] in (UAVState.IDLE.value, UAVState.INIT.value) for i in
                    range(self.UAV_count))
            else:
                return (status == VehicleState.LANDING.value and all(
                    self.agent_status[f"UAV_{group_num}_{i}"] in (UAVState.IDLE.value, UAVState.INIT.value) for i in
                    range(self.UAV_count))
                        and self._is_cluster_delivered(self.group_assigned[group_num]))
        else:
            # UAV can act when it's IDLE (docked and ready) or LANDING (just finished delivery and waiting for next instruction)
            # UAV cannot act when it's DELIVERING or RETURNING
            if status == UAVState.DELIVERING.value or status == UAVState.RETURNING.value:
                return False
            
            if status == UAVState.LANDING.value:
                # UAV just finished a delivery and can take new actions (including returning to vehicle)
                return True
            
            if status == UAVState.IDLE.value:
                # UAV can act if:
                # 1. All vehicles are in LANDING state (not moving)
                # 2. The assigned cluster is not fully delivered
                # 3. If UAV is docked, the vehicle must be at a deployment location
                
                vehicles_landing = all(self.agent_status[f"vehicle_{group_num}_{i}"] == VehicleState.LANDING.value for i in range(self.vehicle_count))
                cluster_not_delivered = not self._is_cluster_delivered(self.group_assigned[group_num])
                
                if not vehicles_landing or not cluster_not_delivered:
                    return False
                
                # If UAV is docked, vehicle should be at a valid deployment location
                if self.isUAVDocked(agent):
                    # UAV can act when vehicle is stopped at a deployment location and cluster is not delivered
                    return True
                    
                # UAV is IDLE but not docked (this shouldn't normally happen, but handle it)
                return True
            
            return False

    def _calc_power_consumption(self, weight, distance):
        """calculate the power consumption of uav

        :param weight: the weight the uav carries
        :param distance: the distance the uav moves
        :return: the power consumption of uav
        """
        return weight * distance * self.power_coefficient
    
    @staticmethod
    def _get_agent_group(agent):
        """get the group number of the agent

        :param agent: the name of the agent
        :return: the group number and agent number of the agent in that group
        """
        return map(int, re.findall(r"\d+", agent))
    
    @staticmethod
    def _calc_distance(start, end):
        return np.linalg.norm(np.array(start) - np.array(end))

    def _k_means_cluster(self):
        """K-means clustering for the nodes(tasks)
        """
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.cluster_number, random_state=0)
            self.kmeans.fit(self.nodes_location)

        elements = []
        for cluster in range(self.kmeans.n_clusters):
            cluster_elements = self.nodes_location[self.kmeans.labels_ == cluster]
            elements.append(np.array(cluster_elements))
        self.infos["cluster_elements"] = copy(elements)
        self.infos["cluster_centers"] = copy(self.kmeans.cluster_centers_)

        mask = {}
        for index, x in enumerate(self.kmeans.labels_):
            if x not in mask:
                mask[x] = []
            mask[x].append(index)
        self.infos["cluster_mask"] = copy(mask)

        center_node = {}
        for x in range(self.kmeans.n_clusters):
            kmeans_center = self.kmeans.cluster_centers_[x]
            # center_node[x] = self.nodes_location[np.argmin(np.linalg.norm(self.nodes_location - kmeans_center, axis=1))]
            # calc the index of the customer node which is closest to the center of the cluster
            min_distance = float("inf")
            min_index = None
            for index, node in enumerate(self.nodes_location):
                distance = self._calc_distance(kmeans_center, node)
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
            center_node[x] = min_index

        self.infos["center_node"] = copy(center_node)

    def _get_k_means_cluster(self, coordinates):
        """get the cluster number of the coordinates

        :param coordinates: the coordinates of the node
        :return: the cluster number of the coordinates
        """
        if self.kmeans is None:
            return None
        new_coordinate = np.array([coordinates])
        return self.kmeans.predict(new_coordinate)[0]
    
    def _check_feasible(self):
        if self.kmeans is None or self.nodes_location is None:
            return False
        
        for i in range(self.cluster_number):
            center = self.infos["center_node"][i]
            print("center", center)
            nodes = self.infos["cluster_mask"][i]
            print("nodes", nodes)

            for node in nodes:
                weight = self.nodes_weight[node]
                dist = self._calc_distance(self.nodes_location[center], self.nodes_location[node])
                rtt_power = self._calc_power_consumption(weight + self.traffic_manager.getUAVWeight(), dist) + \
                    self._calc_power_consumption(self.traffic_manager.getUAVWeight(), dist) # go to the node and back to the truck
                
                print("rtt_power", rtt_power)
                if self.uav_power - rtt_power < 0:
                    return False
        return True
    
    def _is_cluster_delivered(self, cluster):
        """check if the cluster is delivered

        :param cluster: the cluster number
        :return: if the cluster is delivered
        """
        if cluster is None:
            return False
        delivered_or_assigned = np.logical_or(
            self.node_mask[self.kmeans.labels_ == cluster] == PackageState.DELIVERED.value,
            self.node_mask[self.kmeans.labels_ == cluster] == PackageState.ASSIGNED.value
        )
        return np.all(delivered_or_assigned)

    def clearDecisions(self):
        """Clear the decisions for the next time step.
        """
        pass

    def _updateStateInfo(self):
        """Update the state information for the entities.
        """
        # 1. 获取当前时刻的vehicle和UAV的id
        all_vehicle_ids_set = set(self.vehicles.keys())
        all_UAV_ids_set = set(self.UAVs.keys())
        # TODO: 在研究好需要存储的状态信息后，再实现

    def _updateAIModels(self):
        """Update the AI models. Not training the AI models, just updating the AI models when Federated Learning, Transfer Learning, in new regions, etc.
        """
        for node_id, model_dict in self.update_AI_models.items():
            node = self._getNodeById(node_id)
            for model_name, model in model_dict.items():
                node.updateAIModel(model_name, model)

    def _getNodeIdxById(self, node_id):
        """Get the node index by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            int: The index of the node.
        """
        if node_id in self.vehicles:
            return self.vehicle_ids_as_index.index(node_id)
        elif node_id in self.UAVs:
            return self.UAV_ids_as_index.index(node_id)
        else:
            return -1

    def _getNodeTypeById(self, node_id):
        """Get the node type by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            str: The type of the node. 'V' for vehicle, 'U' for UAV,
        """
        if node_id in self.vehicles:
            return 'V'
        elif node_id in self.UAVs:
            return 'U'
        else:
            return None

    def _getNodeById(self, node_id):
        """Get the node by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            SimpleNode: The node.
        """
        node = self.UAVs.get(node_id, None)
        if node is None:
            node = self.vehicles.get(node_id, None)
        return node

    def _updateTask(self):
        """Update and generate the task for the entities. 
        """
        pass

    def getTaskNodeNumByType(self, node_type):
        pass

    def getVehicleIds(self):
        """Get the vehicle ids.

        Returns:
            list: The vehicle ids.
        """
        return list(self.vehicles.keys())

    def getUAVIds(self):
        """Get the list of all UAV IDs currently in the simulation.
        
        Returns:
            list: List of UAV IDs.
        """
        return list(self.UAVs.keys())


    def getVehicleById(self, id):
        """Get the vehicle by the given id.

        Args:
            id (str): The id of the vehicle.

        Returns:
            vehicle: The vehicle.
        """
        return self.vehicles[id]

    def getUAVById(self, id):
        """Get the UAV by the given id.

        Args:
            id (str): The id of the UAV.

        Returns:
            UAV: The UAV.
        """
        return self.UAVs[id]

    def getUAVBatteryLevel(self, uav_id):
        """Get the battery level of a UAV as a percentage.
        
        Args:
            uav_id (str): The ID of the UAV.
            
        Returns:
            float: The battery level as a percentage (0-100), or None if UAV not found.
        """
        if uav_id not in self.UAVs:
            return None
            
        return self.UAVs[uav_id].getBatteryLevel()
        
    def getUAVBatteryInfo(self, uav_id):
        """Get detailed battery information for a UAV.
        
        Args:
            uav_id (str): The ID of the UAV.
            
        Returns:
            dict: Battery information including capacity, current energy, and consumption rate,
                  or None if UAV not found.
        """
        if uav_id not in self.UAVs:
            return None
            
        uav = self.UAVs[uav_id]
        return {
            "battery_capacity": uav._battery_capacity,
            "current_energy": uav._current_energy,
            "energy_consumption_rate": uav._energy_consumption_rate,
            "battery_percentage": uav.getBatteryLevel()
        }
        
    def getAllUAVsBatteryLevels(self):
        """Get battery levels for all UAVs.
        
        Returns:
            dict: Dictionary mapping UAV IDs to battery levels (as percentages).
        """
        battery_levels = {}
        for uav_id, uav in self.UAVs.items():
            battery_levels[uav_id] = uav.getBatteryLevel()
        return battery_levels

    def getAllUAVsStatus(self):
        """Get comprehensive status information for all UAVs.
        
        Returns:
            dict: Dictionary mapping UAV IDs to their status information.
        """
        uav_status = {}
        for uav_id in self.UAVs.keys():
            uav_status[uav_id] = {
                'capacity': self.traffic_manager.getCurrentUAVCapacity(uav_id),
                'power': self.traffic_manager.getCurrentUAVPower(uav_id),
                'travel_distance': self.traffic_manager.getCurrentUAVTravelDistance(uav_id),
                'battery_percentage': self.traffic_manager.getBatteryLevel(uav_id),
                'is_docked': self.isUAVDocked(uav_id),
                'position': self.UAVs[uav_id].getPosition2D()
            }
        return uav_status

    def getAllVehiclesStatus(self):
        """Get comprehensive status information for all vehicles.
        
        Returns:
            dict: Dictionary mapping vehicle IDs to their status information.
        """
        vehicle_status = {}
        for vehicle_id in self.vehicles.keys():
            vehicle_status[vehicle_id] = {
                'position': self.vehicles[vehicle_id].getPosition2D(),
                'speed': self.vehicles[vehicle_id].getSpeed(),
                'onboard_uavs': self.traffic_manager.getOnBoardUAVs(vehicle_id)
            }
        return vehicle_status

    def _initVehicle(self, vehicle_traffic_info):
        """Initialize the vehicle.

        Args:
            vehicle_traffic_info (dict): The vehicle traffic information.

        Returns:
            vehicle: The vehicle.

        Examples:
            vehicle_traffic_info = {"id": "vehicle_1", "position": (0, 0, 0), "speed": 10, "acceleration": 0, "angle": 0}
        """
        vehicle = Vehicle(vehicle_traffic_info['id'], vehicle_traffic_info['position'], vehicle_traffic_info['speed'],
                          vehicle_traffic_info['acceleration'], vehicle_traffic_info['angle'])
        return vehicle

    def _getDistanceBetweenNodes(self, node1, node2):
        """Get the distance between two nodes.

        Args:
            node1 (SimpleNode): The first node.
            node2 (SimpleNode): The second node.

        Returns:
            float: The distance between two nodes.
        """
        return np.linalg.norm(np.array(node1.getPosition()) - np.array(node2.getPosition()))

    def getDistanceBetweenNodesById(self, node_id_1, node_id_2):
        node_1 = self._getNodeById(node_id_1)
        node_2 = self._getNodeById(node_id_2)
        return self._getDistanceBetweenNodes(node_1, node_2)

    def _updateTraffics(self):
        """Update the vehicle traffics.
        """
        self.traffic_manager.stepSimulation()
        vehicle_traffic_infos = self.traffic_manager.getVehicleTrafficInfos()
        UAV_traffic_infos = self.traffic_manager.getUAVTrafficInfos()

        # # Initialize vehicles if first simulation step
        # if self.simulation_time == 0:
        #     for vehicle_id, vehicle_traffic_info in vehicle_traffic_infos.items():
        #         self.vehicles[vehicle_id] = self._initVehicle(vehicle_traffic_info)
                
        #     for UAV_id, UAV_traffic_info in UAV_traffic_infos.items():
        #         self.UAVs[UAV_id] = UAV(UAV_id, UAV_traffic_info['position'], UAV_traffic_info['speed'],
        #                             UAV_traffic_info['acceleration'], UAV_traffic_info['angle'],
        #                             UAV_traffic_info['phi'])
        
        # Update existing vehicles
        for vehicle_id, vehicle_traffic_info in vehicle_traffic_infos.items():
            if vehicle_id in self.vehicles:
                self.vehicles[vehicle_id].update(vehicle_traffic_info, self.simulation_time)
            else:
                # This would only happen if a new vehicle was generated to replace one that left the simulation
                self.vehicles[vehicle_id] = self._initVehicle(vehicle_traffic_info)
                
        # Update existing UAVs
        for UAV_id, UAV_traffic_info in UAV_traffic_infos.items():
            if UAV_id in self.UAVs:
                self.UAVs[UAV_id].update(UAV_traffic_info, self.simulation_time)
            else:
                # This shouldn't happen with fixed UAVs, but handle just in case
                battery_capacity = UAV_traffic_info.get('battery_capacity', 100)
                energy_consumption_rate = UAV_traffic_info.get('energy_consumption_rate', 1)
                self.UAVs[UAV_id] = UAV(UAV_id, UAV_traffic_info['position'], UAV_traffic_info['speed'],
                                    UAV_traffic_info['acceleration'], UAV_traffic_info['angle'],
                                    UAV_traffic_info['phi'], battery_capacity, energy_consumption_rate)

        # Update the index lists
        self.vehicle_ids_as_index = list(self.vehicles.keys())
        self.UAV_ids_as_index = list(self.UAVs.keys())
        
    # UAV docking and takeoff methods
    def dockUAV(self, uav_id, vehicle_id):
        """Dock a UAV to a vehicle.
        
        Args:
            uav_id (str): The ID of the UAV to dock.
            vehicle_id (str): The ID of the vehicle to dock to.
            
        Returns:
            bool: True if docking was successful, False otherwise.
        """
        if uav_id not in self.UAVs or vehicle_id not in self.vehicles:
            return False
            
        # Update UAV object
        self.UAVs[uav_id].dock(vehicle_id)
        
        # Update in traffic manager
        return self.traffic_manager.dockUAV(uav_id, vehicle_id)
    
    def takeOffUAV(self, uav_id, initial_speed=None, initial_angle=None, initial_phi=None):
        """Take off a UAV from its docked vehicle.
        
        Args:
            uav_id (str): The ID of the UAV to take off.
            initial_speed (float, optional): Initial speed for takeoff.
            initial_angle (float, optional): Initial horizontal angle for takeoff.
            initial_phi (float, optional): Initial vertical angle for takeoff.
            
        Returns:
            bool: True if takeoff was successful, False otherwise.
        """
        if uav_id not in self.UAVs:
            return False
            
        # Check if UAV is actually docked
        if not self.isUAVDocked(uav_id):
            return False
            
        # Update UAV object
        default_speed = 20 if initial_speed is None else initial_speed
        default_angle = 0 if initial_angle is None else initial_angle
        default_phi = 0 if initial_phi is None else initial_phi
        
        self.UAVs[uav_id].takeOff(default_speed, default_angle, default_phi)
        
        # Update in traffic manager
        return self.traffic_manager.takeOffUAV(uav_id, initial_speed, initial_angle, initial_phi)
    
    def setUAVDestination(self, uav_id, destination):
        """Set the destination for a UAV to fly to.
        
        Args:
            uav_id (str): The ID of the UAV.
            destination (tuple): The destination coordinates (x, y, z).
            
        Returns:
            bool: True if destination was set successfully, False otherwise.
        """
        if uav_id not in self.UAVs:
            return False
            
        # Check if UAV is not docked
        if self.isUAVDocked(uav_id):
            # Can't set destination for a docked UAV
            return False
            
        # Update UAV object
        self.UAVs[uav_id].setDestination(destination)
        
        # Update in traffic manager
        return self.traffic_manager.setUAVDestination(uav_id, destination)
    
    def isUAVDocked(self, uav_id):
        """Check if a UAV is docked to a vehicle.
        
        Args:
            uav_id (str): The ID of the UAV.
            
        Returns:
            bool: True if UAV is docked, False otherwise.
        """
        if uav_id not in self.UAVs:
            return False
        return self.traffic_manager.isUAVDocked(uav_id)
    
    def getDockedVehicleId(self, uav_id):
        """Get the ID of the vehicle that a UAV is docked to.
        
        Args:
            uav_id (str): The ID of the UAV.
            
        Returns:
            str: The ID of the vehicle, or None if not docked.
        """
        if uav_id not in self.UAVs or not self.isUAVDocked(uav_id):
            return None
            
        return self.traffic_manager.getDockedVehicleId(uav_id)