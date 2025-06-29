import traci
import numpy as np
import random
import pandas as pd
import sumolib
from ..utils.enums import UAVState, VehicleState, PackageState,UAVActionRet

class TrafficManager():
    """The traffic manager class. It manages both vehicle traffic and UAV traffic. It also manipulates the positions of the vehicles, UAVs, RSUs, and cloud servers.
    """

    def __init__(self, config_traffic, traci_connection:traci.connection, sumo_network_xml:str=None):
        """Initialize the traffic manager.

        Args:
            config_traffic (dict): The traffic configuration part of the environment configuration.
        """
        self._config_traffic = config_traffic
        self._net = sumolib.net.readNet(sumo_network_xml)
        self._number_of_vehicles = config_traffic.get("vehicle_count", 1)
        self._x_range = config_traffic.get("x_range", [0, 1000]) # set in airfogsim_env.py according to used area map
        self._y_range = config_traffic.get("y_range", [0, 1000]) # set in airfogsim_env.py according to used area map
        self._nonfly_zone_coordinates = config_traffic.get("nonfly_zone_coordinates", [])
        self._UAV_z_range = config_traffic.get("UAV_z_range", [1, 2])
        self._UAV_speed_range=config_traffic.get("UAV_speed_range",[15,20])
        self._number_of_UAVs = config_traffic.get("UAV_count", 2)
        self._UAV_weight = config_traffic.get("UAV_weight", 1)
        self._UAV_capacity = config_traffic.get("UAV_capacity", 10)
        self._group_num = config_traffic.get("group_num", 1)
        self._distance_threshold = config_traffic.get("distance_threshold", 1)
        
        # UAV battery settings
        self._UAV_battery_capacity = config_traffic.get("UAV_battery_capacity", 100)
        self._UAV_energy_consumption_rate = config_traffic.get("UAV_energy_consumption_rate", 0.1)
        
        # UAV docking offset from vehicle (z-axis only for now)
        self._UAV_docking_z_offset = config_traffic.get("UAV_docking_z_offset", 2.0)  # height above vehicle
        
        # Track UAV docking status
        self._UAV_docking_info = {}  # UAV_id -> {'is_docked': bool, 'vehicle_id': str}

        self._traci_connection = traci_connection
        self._current_time = 0.0

        self._vehicle_infos = {} # vehicle_id -> {position, speed, routeId}
        self._UAV_infos = {} # uav_id -> {position, speed, acceleration, angle, phi}

        self._sumo_route_ids = [] # all route ids in SUMO, further information can be gained by traci_connection.route.getEdges(route_id)
        self._sumo_edges = {} # each edge is a series of lanes in SUMO, edgeId -> [laneId1, laneId2, ...]
        self._sumo_laneIds = [] # all lane ids in SUMO

        self._traffic_interval = config_traffic.get("traffic_interval", 1)
        self._tripinfo = None
        if traci_connection is not None:
            print( traci_connection.simulation.getDeltaT())
            assert traci_connection.simulation.getDeltaT() == self._traffic_interval, "The traffic interval should be the same as the simulation interval."
        else:
            # 从config_traffic的tripinfo中获取tripinfo.csv的路径，读取作为pandas的DataFrame
            tripinfo_path = config_traffic.get("tripinfo", None)
            if tripinfo_path is not None:
                # 只读取columns=['vehicle_id', 'data_timestep', 'vehicle_x', 'vehicle_y', 'vehicle_speed', 'vehicle_angle', 'vehicle_route']
                self._tripinfo = pd.read_csv(tripinfo_path, sep=";", usecols=['vehicle_id', 'data_timestep', 'vehicle_x', 'vehicle_y', 'vehicle_speed', 'vehicle_angle', 'vehicle_route'])
                # dropnan
                self._tripinfo = self._tripinfo.dropna()
            else:
                raise ValueError("The tripinfo path is not set in the config_traffic.")
        
        self._vehicle_id_counter = 0
        self._UAV_id_counter = 0
        self._route_id_counter = 0

        self._grid_width = 50
        self._traffic_mode = config_traffic['traffic_mode']

        self._initialize_map_by_grid()
        self._initialize_edges_and_lanes()
        self._update_route_ids()
        
        # Initialize vehicles and UAVs at the start
        if self._traffic_mode == 'SUMO':
            self._initialize_vehicles()
        self._initialize_UAVs()
        
        # Initially dock UAVs to vehicles (distribute evenly if possible)
        self._initialize_UAV_docking()

    def reset(self, traci_connection = None):
        """Reset the traffic manager.
        """
        self._traci_connection = traci_connection
        self._current_time = 0.0
        self._vehicle_infos = {}
        self._UAV_infos = {}
        self._UAV_docking_info = {}
        self._vehicle_id_counter = 0
        self._UAV_id_counter = 0
        self._route_id_counter = 0
        
        self._initialize_map_by_grid()
        self._initialize_edges_and_lanes()
        self._update_route_ids()
        
        # Re-initialize vehicles and UAVs at reset
        if self._traffic_mode == 'SUMO':
            self._initialize_vehicles()
        self._initialize_UAVs()
        
        # Re-dock UAVs to vehicles
        self._initialize_UAV_docking()

    def getMapIndexByNodeId(self, node_id):
        # row_idx, col_idx = np.where(self._map_by_grid == node_id)
        for row in range(self._map_by_grid.shape[0]):
            for col in range(self._map_by_grid.shape[1]):
                if node_id in self._map_by_grid[row, col]:
                    return row, col
        return None, None
    
    def getVehicleTrafficInfosByMapIndex(self, row, col):
        row = min(max(0, row), self._map_by_grid.shape[0] - 1)
        col = min(max(0, col), self._map_by_grid.shape[1] - 1)
        vehicle_ids = self._map_by_grid[row, col]
        vehicle_infos = self.getVehicleInfoByIds(vehicle_ids)
        return vehicle_infos
    
    def getMapIndexesByTargetPositionAndRange(self, target_position, range):
        row = int((target_position[1] - self._y_range[0]) / self._grid_width)
        col = int((target_position[0] - self._x_range[0]) / self._grid_width)
        row_range = int(range / self._grid_width)
        col_range = int(range / self._grid_width)
        row_start = max(0, row - row_range)
        row_end = min(self._map_by_grid.shape[0], row + row_range + 1)
        col_start = max(0, col - col_range)
        col_end = min(self._map_by_grid.shape[1], col + col_range + 1)
        return row_start, row_end, col_start, col_end

    @property
    def map_by_grid(self):
        return self._map_by_grid.copy()
    
    @property
    def grid_width(self):
        return self._grid_width

    def _initialize_map_by_grid(self):
        """Initialize the map_by_grid matrix. The matrix is used to store the node ids (as list) in each grid. The grid is defined by the grid width. The matrix is by: row1, col1 = y1, x1; row2, col2 = y2, x2 of position (x, y). 
        """
        row_num = int((self._y_range[1] - self._y_range[0]) / self._grid_width)
        col_num = int((self._x_range[1] - self._x_range[0]) / self._grid_width)
        self._map_by_grid = np.empty((row_num, col_num), dtype=object)
        for i in range(row_num):
            for j in range(col_num):
                self._map_by_grid[i, j] = []
    
    def getNumberOfUAVs(self):
        """Get the number of UAVs.

        Returns:
            int: The number of UAVs.
        """
        return len(self._UAV_infos)
    
    def getNumberOfVehicles(self):
        """Get the number of vehicles.

        Returns:
            int: The number of vehicles.
        """
        return len(self._vehicle_infos)


    def _initialize_vehicles(self):
        """Initialize the vehicles with routes starting from (0,0) position.
        """
        if self._traffic_mode == 'SUMO':
            # Find the edge closest to (0,0)
            start_edges = ['A0A1', 'B0A0', 'A0B0']
            # start_edges = ['481595026', '43925952', '1101731479']
            for group_num in range(self._group_num):
                for vehicle_num in range(self._number_of_vehicles):
                    vehicle_id = "vehicle_" + str(group_num) + "_" + str(vehicle_num)
                    print("vehicle_id", vehicle_id)
                    self._vehicle_id_counter += 1
                    
                    # Generate a route starting from the edge near (0,0)
                    route_id = self._generateRouteFromStart(start_edges[group_num])
                    
                    # Add vehicle with the route
                    self._traci_connection.vehicle.add(vehicle_id, route_id)
                    
                    # Get the first lane of the start edge
                    start_lane = self._sumo_edges[start_edges[group_num]][0] if start_edges[group_num] in self._sumo_edges else "0"
                    
                    # Set initial position to (0,0) with all required parameters
                    try:
                        # First try to move the vehicle to the exact position
                        self._traci_connection.vehicle.moveToXY(
                            vehicle_id,      # vehicle ID
                            start_edges[group_num],      # edge ID
                            0,              # lane index (use 0 instead of lane ID)
                            0,              # x position
                            0,              # y position
                            0,              # angle (0 degrees)
                            1               # keepRoute flag
                        )
                    except Exception as e:
                        print(f"Warning: Could not move vehicle {vehicle_id} to exact position: {e}")
                        # If exact positioning fails, let SUMO place the vehicle on the route
                        pass
            
            # Perform one simulation step to place the vehicles in the network
            self._traci_connection.simulationStep()
            
            # Update vehicle infos from SUMO
            vehicle_ids = self.getVehicleIDsList()

            print("vehicle_ids", vehicle_ids)

            self._vehicle_infos = self.getVehicleInfoByIds(vehicle_ids)

    def _generateRouteFromStart(self, start_edge):
        """Generate a route starting from the specified edge.
        
        Args:
            start_edge (str): The starting edge ID.
            
        Returns:
            str: The generated route ID.
        """
        route_id = "gen_veh_route_" + str(self._route_id_counter)
        valid_edges = self.valid_edges
        
        # Try to find a valid route starting from the specified edge
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Randomly select a destination edge
                to_edge = random.choice(valid_edges)
                if to_edge == start_edge:
                    continue
                    
                route = traci.simulation.findRoute(start_edge, to_edge)
                if len(route.edges) > 0:
                    self._traci_connection.route.add(route_id, route.edges)
                    self._route_id_counter += 1
                    return route_id
            except traci.exceptions.TraCIException as e:
                continue
        
        # If no valid route found, create a simple route with just the start edge
        self._traci_connection.route.add(route_id, [start_edge])
        self._route_id_counter += 1
        return route_id

    def setVehicleDestination(self, vehicle_id, destination):
        """Set the destination for a vehicle.
        Args:
            vehicle_id (str): The vehicle id.
            destination (tuple): The destination position, 2D position.
        """
        if self._traffic_mode == 'SUMO':
            current_edge = self._get_vehicle_edge(vehicle_id)
            destination_edge = self._find_nearest_edge(destination, -10, -10)
            route_edges = self._plan_route(current_edge, destination_edge)
            success = self._set_vehicle_route(vehicle_id, route_edges)
            
            if success and route_edges:
                # 设置车辆在目标边上的具体停车位置
                self._set_vehicle_stop_at_destination(vehicle_id, destination_edge, destination)
        else:
            pass

    def _get_vehicle_edge(self, vehicle_id):
        """Find the edge of the vehicle.
        Args:
            vehicle_id (str): The vehicle id.
        Returns:
            str: The edge id.
        """
        try:
            edge_id = self._traci_connection.vehicle.getRoadID(vehicle_id)
            return edge_id
        except Exception as e:
            raise

    def _find_nearest_edge(self, position, deltaX=0, deltaY=0):
        """Find the nearest edge to the given position"""
        result = self._traci_connection.simulation.convertRoad(position[0] + deltaX, position[1] + deltaY)
        # print(f"Nearest edge to position {position} is: {result}")
        
        # Ensure the returned edge is valid
        edge_id = result[0]  # Get edge_id part
        
        # If the edge is not in valid edges list, try to find a valid alternative
        if edge_id not in self.valid_edges and edge_id.lstrip('-') not in self.valid_edges:
            print(f"Warning: Found edge {edge_id} not in valid edges list, trying to find alternative")
            
            # Get lanes near the destination
            lanes = self._traci_connection.lane.getIDList()
            closest_lane = None
            min_dist = float('inf')
            
            for lane in lanes:
                if lane.startswith(":"):  # Skip internal lanes
                    continue
                    
                try:
                    lane_shape = self._traci_connection.lane.getShape(lane)
                    # Calculate distance from lane midpoint to destination
                    mid_point = lane_shape[len(lane_shape)//2]
                    dist = ((mid_point[0] - position[0])**2 + (mid_point[1] - position[1])**2)**0.5
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_lane = lane
                except:
                    continue
            
            if closest_lane:
                edge_id = closest_lane.split("_")[0]  # Extract edge ID from lane ID
                print(f"Found alternative edge: {edge_id}")
                
                # Ensure it's a valid edge
                if edge_id.startswith("-"):
                    edge_id = edge_id[1:]  # Remove negative sign
                
                # Final check if in valid edges list
                if edge_id in self.valid_edges:
                    return edge_id
            
            # If all else fails, use first valid edge as target
            if self.valid_edges:
                print(f"Could not find suitable target edge, using default edge: {self.valid_edges[0]}")
                return self.valid_edges[0]
        
        # If negative, try using version without negative sign
        if edge_id.startswith("-") and edge_id[1:] in self.valid_edges:
            edge_id = edge_id[1:]
            
        return edge_id
    
    def _set_vehicle_route(self, vehicle_id, route_edges):
        """Set the route for a vehicle.
        Args:
            vehicle_id (str): The vehicle id.
            route_edges (list): The route edges.
        """
        try:
            if not route_edges:
                print("Warning: Path is empty, cannot set vehicle route")
                return False
                
            # Set vehicle route
            print(f"Setting vehicle {vehicle_id} route, containing {len(route_edges)} edges")
            self._traci_connection.vehicle.setRoute(vehicle_id, route_edges)
            print(f"Successfully set vehicle {vehicle_id} route")
            return True
        except Exception as e:
            print(f"Error on setting vehicle route: {e}")
            return False

    def _set_vehicle_stop_at_destination(self, vehicle_id, destination_edge, destination):
        """Set vehicle to stop at a specific position on the destination edge.
        Args:
            vehicle_id (str): The vehicle id.
            destination_edge (str): The destination edge id.
            destination (tuple): The target destination position (x, y).
        """
        try:
            # 获取目标边的形状信息
            if destination_edge not in self._sumo_edges:
                print(f"Warning: Destination edge {destination_edge} not found")
                return False
            
            # 获取第一个车道ID
            lanes = self._sumo_edges[destination_edge]
            if not lanes:
                print(f"Warning: No lanes found for edge {destination_edge}")
                return False
            
            target_lane = lanes[0]  # 使用第一个车道
            
            # 获取车道的形状
            lane_shape = self._traci_connection.lane.getShape(target_lane)
            if not lane_shape:
                print(f"Warning: Cannot get shape for lane {target_lane}")
                return False
            
            # 计算目标位置在车道上的位置
            lane_position = self._calculate_position_on_lane(destination, lane_shape)
            
            # 获取车道索引（从边ID中提取）
            lane_index = 0  # 默认使用第一个车道
            if '_' in target_lane:
                try:
                    lane_index = int(target_lane.split('_')[-1])
                except:
                    lane_index = 0
            
            # 设置车辆在目标位置停车
            #print(f"Setting vehicle {vehicle_id} to stop at position {lane_position} on edge {destination_edge}")
            self._traci_connection.vehicle.setStop(
                vehicle_id,         # 车辆ID
                destination_edge,   # 边ID
                lane_position,      # 在边上的位置
                lane_index,         # 车道索引
                30.0,              # 停车时长（秒）
                1                   # 停车标志（1=停车）
            )
            
            return True
            
        except Exception as e:
            print(f"Error setting vehicle stop at destination: {e}")
            return False
    
    def _calculate_position_on_lane(self, target_position, lane_shape):
        """Calculate the position along a lane that is closest to the target position.
        Args:
            target_position (tuple): Target position (x, y).
            lane_shape (list): List of coordinate points defining the lane shape.
        Returns:
            float: Position along the lane (0 to lane_length).
        """
        if not lane_shape or len(lane_shape) < 2:
            return 0.0
        
        min_distance = float('inf')
        best_position = 0.0
        total_length = 0.0
        
        target_x, target_y = target_position[0], target_position[1]
        
        # 遍历车道形状的每个线段
        for i in range(len(lane_shape) - 1):
            p1 = lane_shape[i]
            p2 = lane_shape[i + 1]
            
            # 计算线段长度
            segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 在线段上找到距离目标点最近的点
            if segment_length > 0:
                # 参数化线段: P = P1 + t * (P2 - P1), t ∈ [0, 1]
                # 找到使距离最小的t值
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                t = max(0, min(1, ((target_x - p1[0]) * dx + (target_y - p1[1]) * dy) / (dx**2 + dy**2)))
                
                # 计算线段上最近点的坐标
                closest_x = p1[0] + t * dx
                closest_y = p1[1] + t * dy
                
                # 计算距离
                distance = np.sqrt((target_x - closest_x)**2 + (target_y - closest_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_position = total_length + t * segment_length
            
            total_length += segment_length
        
        # 确保位置在有效范围内
        return max(0.0, min(best_position, total_length - 1.0))
        

    def _plan_route(self, from_edge, to_edge):
        """Plan a route from the starting edge to the target edge.
        Args:
            from_edge (str): The starting edge id.
            to_edge (str): The target edge id.
        Returns:
            list: The route edges.
        """
        try:
            # Ensure both starting and target edges are valid
            if from_edge not in self._traci_connection.edge.getIDList():
                print(f"Warning: Starting edge {from_edge} not in network")
                if from_edge.startswith("-") and from_edge[1:] in self.valid_edges:
                    from_edge = from_edge[1:]
                    print(f"Trying alternative starting edge: {from_edge}")
                else:
                    from_edge = self.valid_edges[0]
                    print(f"Using default starting edge: {from_edge}")
            
            if to_edge not in self._traci_connection.edge.getIDList():
                print(f"Warning: Target edge {to_edge} not in network")
                to_edge = self.valid_edges[-1]
                print(f"Using default target edge: {to_edge}")
            
            # Try multiple times to find a valid path
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    route = self._traci_connection.simulation.findRoute(from_edge, to_edge)
                    # Check if a valid path was found
                    if len(route.edges) > 0:
                        print(f"Planning route from {from_edge} to {to_edge}")
                        print(f"Path length: {route.length:.2f} meters, cost: {route.cost:.2f}")
                        print(f"Found a valid path, containing {len(route.edges)} edges:")
                        for edge in route.edges:
                            print(f" - {edge}")
                        return route.edges
                    else:
                        print(f"Attempt {attempt+1}: No path found from {from_edge} to {to_edge}")
                        
                        # If no path is found, randomly select a new target edge
                        if attempt < max_attempts - 1:  # Not the last attempt
                            to_edge = random.choice(self.valid_edges)
                            print(f"Trying new target edge: {to_edge}")
                except traci.exceptions.TraCIException as e:
                    print(f"Error on attempt {attempt+1}: {e}")
                    if attempt < max_attempts - 1:  # Not the last attempt
                        # Randomly select new edges
                        from_edge, to_edge = random.sample(self.valid_edges, 2)
                        print(f"Trying new edge pair: {from_edge} -> {to_edge}")
            
            print("No valid path found, using a simple path")
            # If all attempts fail, return a simple path containing a single edge
            return [self.valid_edges[0]]
        except Exception as e:
            print(f"Error on planning route: {e}")
            # If an error occurs, return a simple path containing a single edge
            return [self.valid_edges[0]] if self.valid_edges else []
        

    def _arrive_near_destination(self, vehicle_id, destination, threshold=10.0):
        """Decide whether the vehicle has arrived near the destination.
        Args:
            vehicle_id (str): The vehicle id.
            destination (tuple): The destination position (x, y).
            threshold (float): Distance threshold to consider as arrived.
        """
        if self._traffic_mode == 'SUMO':
            try:
                # 获取车辆当前位置
                current_position = self._traci_connection.vehicle.getPosition(vehicle_id)
                
                # 计算与目标位置的距离
                distance = np.sqrt((current_position[0] - destination[0])**2 + 
                                 (current_position[1] - destination[1])**2)
                
                return distance <= threshold
            except:
                # 如果获取位置失败，回退到原来的边比较方法
                current_edge = self._get_vehicle_edge(vehicle_id)
                destination_edge = self._find_nearest_edge(destination, -10, -10)
                return current_edge == destination_edge
        else:
            pass

    def hasVehicleArrivedAtDestination(self, vehicle_id, destination, threshold=10.0):
        """Check if a vehicle has arrived at its destination.
        Args:
            vehicle_id (str): The vehicle id.
            destination (tuple): The destination position (x, y).
            threshold (float): Distance threshold to consider as arrived. Defaults to 10.0.
        Returns:
            bool: True if the vehicle has arrived, False otherwise.
        """
        return self._arrive_near_destination(vehicle_id, destination, threshold)

    def _initialize_UAVs(self):
        """Initialize the UAV information with random positions in the given range.
        """
        for group_num in range(self._group_num):
            for uav_num in range(self._number_of_UAVs):
                UAV_id = "UAV_" + str(group_num) + "_" + str(uav_num)
                self._UAV_id_counter += 1
                position = (random.uniform(self._x_range[0], self._x_range[1]), 
                        random.uniform(self._y_range[0], self._y_range[1]), 
                        random.uniform(self._UAV_z_range[0], self._UAV_z_range[1]))
                speed = random.uniform(self._UAV_speed_range[0], self._UAV_speed_range[1])
                angle = random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2π
                phi = random.uniform(-np.pi/4, np.pi/4)  # Random vertical angle
                
                self._UAV_infos[UAV_id] = {
                    "position": position,
                    "speed": speed,
                    "last_speed": speed,
                    "acceleration": 0,
                    "angle": angle,
                    "phi": phi,
                    "is_docked": False,
                    "is_hovering": False,
                    "hover_speed_before": None,
                    "hover_destination_before": None,
                    "docked_vehicle_id": None,
                    "destination": None,
                    "battery_capacity": self._UAV_battery_capacity,
                    "current_energy": self._UAV_battery_capacity,  # Start with full battery
                    "energy_consumption_rate": self._UAV_energy_consumption_rate,
                    "capacity": self._UAV_capacity,
                    "travel_distance_2d": 0 # 2D travel distance, change only when the uav is flying(no docking or hovering), reset to 0 when docked
                }
                
                row = int((position[1] - self._y_range[0]) / self._grid_width)
                col = int((position[0] - self._x_range[0]) / self._grid_width)
                if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                    self._map_by_grid[row, col].append(UAV_id)

    def _initialize_edges_and_lanes(self):
        """Initialize the edges and lanes information."""
        if self._net:
            self._sumo_edges = {}
            self._sumo_laneIds = []
            nodes = self._net.getNodes()
            self._sumo_junction_positions = {}
            for node in nodes:
                node_id = node.getID()
                position = (node.getCoord()[0], node.getCoord()[1], 0)
                self._sumo_junction_positions[node_id] = position
            
            # 获取所有的 lane IDs
            for edge in self._net.getEdges():
                for lane in edge.getLanes():
                    lane_id = lane.getID()
                    edge_id = edge.getID()
                    if edge_id not in self._sumo_edges:
                        self._sumo_edges[edge_id] = []
                    self._sumo_edges[edge_id].append(lane_id)
                    self._sumo_laneIds.append(lane_id)

            valid_edges = []
            edges = list(self._sumo_edges.keys())
            self.all_allowed_classes = set()

            for edge in edges:
                lanes = self._sumo_edges[edge]
                for lane_id in lanes:
                    lane = self._net.getLane(lane_id)
                    allowed_classes = lane.getPermissions()
                    self.all_allowed_classes.update(allowed_classes)
                    if len(allowed_classes) == 0 or 'passenger' in allowed_classes:
                        valid_edges.append(edge)
                        break

            self.valid_edges = valid_edges
        

    def _update_route_ids(self):
        """Update the route information generated by SUMO.
        """
        if self._traffic_mode == 'SUMO':
            route_ids = self._traci_connection.route.getIDList()
            self._sumo_route_ids = route_ids

    def _generateRandomRoute(self):
        """Generate a random route id.

        Returns:
            str: The route id.
        """
        route_id = "gen_veh_route_" + str(self._route_id_counter)
        valid_edges = self.valid_edges
        while True:
            try:
                from_edge, to_edge = random.sample(valid_edges, 2) 
                route = traci.simulation.findRoute(from_edge, to_edge)
                while len(route.edges) == 0:
                    from_edge, to_edge = random.sample(valid_edges, 2)
                    route = traci.simulation.findRoute(from_edge, to_edge)
                break
            except traci.exceptions.TraCIException as e:
                pass
            
        self._traci_connection.route.add(route_id, route.edges)
        self._route_id_counter += 1
        return route_id
    def setVehicleRoute(self, vehicle_id, route_id):
        """Set the route for a vehicle.
        """
        pass
    
    def updateVehicleMobilityPatterns(self, vehicle_mobility_patterns):
        """Update the vehicle mobility patterns.

        Args:
            vehicle_mobility_patterns (dict): The vehicle mobility patterns. The key is vehicle id, and the value is the mobility pattern={angle, speed}
        """
        for vehicle_id, mobility_pattern in vehicle_mobility_patterns.items():
            self._traci_connection.vehicle.setSpeed(vehicle_id, mobility_pattern["speed"])

    def _updateUAVMobilityPatternById(self, UAV_id, mobility_pattern):
        """Update the UAV mobility pattern by the UAV id.

        Args:
            UAV_id (str): The UAV id.
            mobility_pattern (dict): The mobility pattern={angle, phi, speed}
        """
        assert UAV_id in self._UAV_infos, "The UAV id should be in the UAV information."
        self._UAV_infos[UAV_id]["speed"] = mobility_pattern["speed"]
        self._UAV_infos[UAV_id]["angle"] = mobility_pattern["angle"]
        self._UAV_infos[UAV_id]["phi"] = mobility_pattern["phi"]

    def updateUAVMobilityPatterns(self, UAV_mobility_patterns):
        """Update the UAV mobility patterns.

        Args:
            UAV_mobility_patterns (dict): The UAV mobility patterns. The key is UAV id, and the value is the mobility pattern={angle, phi, speed}
        """
        for UAV_id, mobility_pattern in UAV_mobility_patterns.items():
            self._updateUAVMobilityPatternById(UAV_id, mobility_pattern)

    def updateCurrentTime(self):
        """Update the current time.

        Returns:
            float: The updated current time.
        """
        if self._traffic_mode == 'SUMO':
            return self._traci_connection.simulation.getTime()
        else:
            # 把self._tripinfo中current_time之前的数据删除
            # self._tripinfo = self._tripinfo[self._tripinfo['data_timestep']>=self._current_time]
            tmp_time = self._current_time + self._traffic_interval
            # 保证tmp_time mod self._traffic_interval == 0
            tmp_time = round(tmp_time / self._traffic_interval) 
            tmp_time = tmp_time * self._traffic_interval
            return tmp_time

    def getVehicleIDsList(self):
        """Get the vehicle ids list.

        Returns:
            list: The vehicle ids list.
        """
        if self._traffic_mode == 'SUMO':
            return self._traci_connection.vehicle.getIDList()
        else:
            # 根据当前的时隙，从tripinfo中获取当前时隙的车辆信息
            current_time = self._current_time
            # tripinfo是pd.DataFrame，可以直接使用pandas的查询功能,date_timestep在current_time-traffic_interval到current_time之间的车辆
            vehicle_ids = self._tripinfo[(self._tripinfo['data_timestep']>current_time-self._traffic_interval) & (self._tripinfo['data_timestep']<=current_time)]['vehicle_id'].tolist()
            return vehicle_ids
        
    def getVehicleInfoByIds(self, vehicle_ids):
        # {"position": position3d, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
        if self._traffic_mode == 'SUMO':
            vehicle_infos = {}
            for vehicle_id in vehicle_ids:
                position = self._traci_connection.vehicle.getPosition(vehicle_id)
                speed = self._traci_connection.vehicle.getSpeed(vehicle_id)
                acceleration = self._traci_connection.vehicle.getAcceleration(vehicle_id)
                angle = self._traci_connection.vehicle.getAngle(vehicle_id)
                route_id = self._traci_connection.vehicle.getRouteID(vehicle_id)
                position3d = (position[0], position[1], 0)
                vehicle_infos[vehicle_id] = {"position": position3d, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
            return vehicle_infos
        else:
            # 从pd中批量获取车辆信息
            cur_time_trip_info = self._tripinfo[(self._tripinfo['data_timestep']>self._current_time-self._traffic_interval) & (self._tripinfo['data_timestep']<=self._current_time)]
            pd_vehicle_infos = cur_time_trip_info[cur_time_trip_info['vehicle_id'].isin(vehicle_ids)]
            vehicle_infos = {}
            for idx, vehicle_info in pd_vehicle_infos.iterrows():
                position = (vehicle_info['vehicle_x'], vehicle_info['vehicle_y'], 0)
                speed = vehicle_info['vehicle_speed']
                acceleration = 0
                angle = vehicle_info['vehicle_angle']
                route_id = vehicle_info['vehicle_route']
                vehicle_id = vehicle_info['vehicle_id']
                vehicle_infos[vehicle_id] = {"position": position, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
            return vehicle_infos

    def stepSimulation(self):
        """Step the simulation for one step. Update positions of existing vehicles and UAVs without generating new ones.
        """
        if self._traffic_mode == 'SUMO':
            # Simply perform a simulation step without adding new vehicles
            self._traci_connection.simulationStep()
            # vehicles will be updated by sumo. (Vehicles which are out of map will be cleared automatically by sumo)
            vehicle_ids = self.getVehicleIDsList()
        else:
            # 从tripinfo中获取当前时间的车辆信息
            vehicle_ids = self.getVehicleIDsList()
            
        self._current_time = self.updateCurrentTime()
        self._vehicle_infos = self.getVehicleInfoByIds(vehicle_ids)

        # Update UAV positions, handling both docked and flying UAVs
        for UAV_id in self._UAV_infos:
            UAV_info = self._UAV_infos[UAV_id]
            
            # Case 1: UAV is docked to a vehicle
            if UAV_info.get("is_docked", False) and UAV_info.get("docked_vehicle_id") in self._vehicle_infos:
                vehicle_id = UAV_info["docked_vehicle_id"]
                vehicle_position = self._vehicle_infos[vehicle_id]["position"]
                
                # Update UAV position to match vehicle position with z-offset
                UAV_info["position"] = (
                    vehicle_position[0],
                    vehicle_position[1],
                    vehicle_position[2] + self._UAV_docking_z_offset
                )
                
                # Docked UAVs don't move independently and don't consume energy
                UAV_info["speed"] = 0
                UAV_info["acceleration"] = 0
                UAV_info["last_speed"] = 0
                
                # Recharge UAV battery when docking
                self._UAV_infos[UAV_id]["current_energy"] = self._UAV_infos[UAV_id]["battery_capacity"]
                
                # Match vehicle angle for visual consistency
                UAV_info["angle"] = self._vehicle_infos[vehicle_id].get("angle", 0)
                UAV_info["phi"] = 0

                # make sure all the other attributes are correct(just in case)
                UAV_info["is_hovering"] = False
                UAV_info["travel_distance_2d"] = 0
                UAV_info["destination"] = None
                UAV_info["capacity"] = self._UAV_capacity
                
            # Case 2: UAV is hovering at a fixed position
            elif UAV_info.get("is_hovering", False):
                # 确保悬停状态在 traffic info 中正确设置
                UAV_info["is_hovering"] = True
                UAV_info["speed"] = 0
                UAV_info["acceleration"] = 0
                
                # Hovering still consumes energy
                self._consumeUAVEnergy(UAV_id)
                
            # Case 3: UAV is flying independently
            else:
                org_position = UAV_info["position"]
                speed = UAV_info.get("speed", 0)
                last_speed = UAV_info.get("last_speed", 0)
                acceleration = (speed - last_speed) / self._traffic_interval
                UAV_info["acceleration"] = acceleration
                UAV_info["last_speed"] = speed
                angle = UAV_info.get("angle", 0)
                phi = UAV_info.get("phi", 0)
                
                # Flying consumes energy
                enough_energy = self._consumeUAVEnergy(UAV_id)
                
                # If no energy left, UAV cannot continue to fly
                if not enough_energy:
                    UAV_info["is_hovering"] = True
                    UAV_info["speed"] = 0
                    UAV_info["acceleration"] = 0
                    print(f"UAV {UAV_id} has depleted its battery and is now hovering in place!")
                    continue
                
                # If UAV has a destination, adjust angle to fly towards it
                destination = UAV_info.get("destination")
                if destination is not None and speed > 0:
                    # Calculate direction vector to destination
                    dx = destination[0] - org_position[0]
                    dy = destination[1] - org_position[1]
                    dz = destination[2] - org_position[2]
                    
                    # Calculate horizontal angle (in radians)
                    if dx != 0 or dy != 0:
                        new_angle = np.arctan2(dy, dx)
                        UAV_info["angle"] = new_angle
                        angle = new_angle
                    
                    # Calculate vertical angle (phi)
                    horizontal_dist = np.sqrt(dx**2 + dy**2)
                    if horizontal_dist > 0:
                        new_phi = np.arctan2(dz, horizontal_dist)
                        UAV_info["phi"] = new_phi
                        phi = new_phi
                    
                    # Check if UAV has reached destination (within threshold)
                    distance_to_dest = np.sqrt(dx**2 + dy**2 + dz**2)
                    if distance_to_dest < self._distance_threshold:
                        # UAV has reached destination
                        UAV_info["destination"] = None
                        UAV_info["speed"] = 0
                        # Automatically hover when reaching destination
                        UAV_info["is_hovering"] = True
                        print(f"UAV {UAV_id} reached destination and is now hovering")
                        continue
                
                # Update position based on current speed and angles
                new_position = (
                    org_position[0] + speed * np.cos(angle) * np.cos(phi) * self._traffic_interval,
                    org_position[1] + speed * np.sin(angle) * np.cos(phi) * self._traffic_interval,
                    org_position[2] + speed * np.sin(phi) * self._traffic_interval
                )
                
                # Convert to float values
                new_position = [float(i) for i in new_position]
                
                # Calculate 2D travel distance and update
                distance_moved_2d = np.sqrt((new_position[0] - org_position[0])**2 + (new_position[1] - org_position[1])**2)
                UAV_info["travel_distance_2d"] += distance_moved_2d
                
                UAV_info["position"] = new_position
        
        self._update_route_ids()
        self._update_map_by_grid()

    def getCurrentUAVTravelDistance(self, UAV_id):
        """Get the current travel distance of a UAV.
        """
        return self._UAV_infos[UAV_id]["travel_distance_2d"]

    def getCurrentUAVPower(self, UAV_id):
        """Get the current power of a UAV.
        """
        return self._UAV_infos[UAV_id]["current_energy"]

    def getCurrentUAVCapacity(self, UAV_id):
        """Get the current capacity of a UAV.
        """
        return self._UAV_infos[UAV_id]["capacity"]

    def changeUAVCapacity(self, UAV_id, capacity):
        self._UAV_infos[UAV_id]["capacity"] = capacity
        return True

    def _update_map_by_grid(self):
        self._map_by_grid = np.empty((self._map_by_grid.shape[0], self._map_by_grid.shape[1]), dtype=object)
        for i in range(self._map_by_grid.shape[0]):
            for j in range(self._map_by_grid.shape[1]):
                self._map_by_grid[i, j] = []
        for vehicle_id, vehicle_info in self._vehicle_infos.items():
            position = vehicle_info["position"]
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(vehicle_id)
        for UAV_id, UAV_info in self._UAV_infos.items():
            position = UAV_info["position"]
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(UAV_id)

    def getVehicleTrafficInfos(self):
        """Get the vehicle traffics at the given simulation time.

        Returns:
            dict: The vehicle traffics, including the vehicle id, position, speed, angle, acceleration, and current routeId.
        """
        return self._vehicle_infos
    
    def getUAVTrafficInfos(self):
        """Get the UAV traffics at the given simulation time. The trajectory of the UAVs is controlled by their missions

        Returns:
            dict: The UAV traffics, including the UAV id, position, acceleration, speed, angle, and phi.
        """
        return self._UAV_infos

    def getNewVehicleIds(self):
        """Get vehicle ids which is added in latest timeslot.

        Returns:
            list: The Id list of vehicles.
        """
        return self._new_added_vehicle_ids
    
    def getCurrentTime(self):
        """Get the current simulation time.

        Returns:
            float: The current simulation time (in seconds).
        """
        # return self._traci_connection.simulation.getTime()
        return self._current_time

    def checkIsRemovingByUAVId(self, UAV_id):
        # Modified to check if UAV is in the simulation, not whether it's moving
        return UAV_id in self._UAV_infos
    
    def isUAVDocked(self, UAV_id):
        """Check if a UAV is docked to a vehicle.
        
        Args:
            UAV_id (str): The ID of the UAV.
            
        Returns:
            bool: True if UAV is docked, False otherwise.
        """
        if UAV_id in self._UAV_infos:
            return self._UAV_infos[UAV_id].get("is_docked", False)
        return False
    
    def getDockedVehicleId(self, UAV_id):
        """Get the ID of the vehicle that a UAV is docked to.
        
        Args:
            UAV_id (str): The ID of the UAV.
            
        Returns:
            str: The ID of the vehicle, or None if not docked.
        """
        if UAV_id in self._UAV_infos and self._UAV_infos[UAV_id].get("is_docked", False):
            return self._UAV_infos[UAV_id].get("docked_vehicle_id")
        return None
    
    def getOnBoardUAVs(self, vehicle_id):
        """Get the UAVs on board of a vehicle.
        """
        uavs_on_board = []
        for uav_id, uav_info in self._UAV_infos.items():
            if uav_info.get("is_docked", False) and uav_info.get("docked_vehicle_id") == vehicle_id:
                uavs_on_board.append(uav_id)
        return uavs_on_board

    def getConfig(self,name):
        return self._config_traffic.get(name,None)
    
    def getNodePositionById(self, id):
        if id in self._vehicle_infos:
            return self._vehicle_infos[id]["position"]
        elif id in self._UAV_infos:
            return self._UAV_infos[id]["position"]
        return None
    
    def getAllJunctionPositions(self):
        return list(self._sumo_junction_positions.values())
    
    def getNonFlyZones(self):
        return self._nonfly_zone_coordinates.copy()
    
    def getUAVWeight(self):
        return self._UAV_weight

    def _initialize_UAV_docking(self):
        """Initialize UAV docking by distributing UAVs to vehicles within the same group."""
        # Group vehicles by their group number
        vehicles_by_group = {}
        for vehicle_id in self._vehicle_infos.keys():
            # Extract group number from vehicle_id (format: "vehicle_groupNum_vehicleNum")
            parts = vehicle_id.split('_')
            if len(parts) >= 3:
                group_num = parts[1]
                if group_num not in vehicles_by_group:
                    vehicles_by_group[group_num] = []
                vehicles_by_group[group_num].append(vehicle_id)
        
        # Group UAVs by their group number
        uavs_by_group = {}
        for uav_id in self._UAV_infos.keys():
            # Extract group number from UAV_id (format: "UAV_groupNum_uavNum")
            parts = uav_id.split('_')
            if len(parts) >= 3:
                group_num = parts[1]
                if group_num not in uavs_by_group:
                    uavs_by_group[group_num] = []
                uavs_by_group[group_num].append(uav_id)
        
        # For each group, dock UAVs to vehicles in the same group
        for group_num in uavs_by_group.keys():
            # Skip if no vehicles in this group
            if group_num not in vehicles_by_group or len(vehicles_by_group[group_num]) == 0:
                continue
                
            # Get UAVs and vehicles for this group
            group_uavs = uavs_by_group[group_num]
            group_vehicles = vehicles_by_group[group_num]
            
            # Distribute UAVs evenly among vehicles in the same group
            # If more UAVs than vehicles, some vehicles will have multiple UAVs
            for i, uav_id in enumerate(group_uavs):
                # Assign to vehicle within the same group (cycling through vehicle list if needed)
                vehicle_idx = i % len(group_vehicles)
                vehicle_id = group_vehicles[vehicle_idx]
                
                # Set UAV as docked to this vehicle
                self.dockUAV(uav_id, vehicle_id)

    def dockUAV(self, uav_id, vehicle_id):
        """Dock a UAV to a vehicle.
        
        Args:
            uav_id (str): The ID of the UAV to dock.
            vehicle_id (str): The ID of the vehicle to dock to.
        
        Returns:
            bool: True if docking successful, False otherwise.
        """
        if uav_id not in self._UAV_infos or vehicle_id not in self._vehicle_infos:
            return False
        
        # Update UAV info with docking status
        self._UAV_infos[uav_id]["is_docked"] = True
        self._UAV_infos[uav_id]["docked_vehicle_id"] = vehicle_id
        self._UAV_infos[uav_id]["speed"] = 0
        self._UAV_infos[uav_id]["acceleration"] = 0
        
        # Recharge UAV battery when docking
        self._UAV_infos[uav_id]["current_energy"] = self._UAV_infos[uav_id]["battery_capacity"]

        # refresh the UAV's travel distance
        self._UAV_infos[uav_id]["travel_distance_2d"] = 0

        # refresh the UAV's current capacity, set to the maximum capacity
        self._UAV_infos[uav_id]["capacity"] = self._UAV_capacity
        
        # Update UAV position to match vehicle position with z-offset
        vehicle_position = self._vehicle_infos[vehicle_id]["position"]
        uav_position = (
            vehicle_position[0],
            vehicle_position[1],
            vehicle_position[2] + self._UAV_docking_z_offset
        )
        self._UAV_infos[uav_id]["position"] = uav_position
        
        # Track docking info
        self._UAV_docking_info[uav_id] = {
            "is_docked": True,
            "vehicle_id": vehicle_id
        }
        
        return True
    
    def takeOffUAV(self, uav_id, initial_speed=None, initial_angle=None, initial_phi=None):
        """Take off a UAV from its docked vehicle.
        
        Args:
            uav_id (str): The ID of the UAV to take off.
            initial_speed (float, optional): Initial speed for takeoff.
            initial_angle (float, optional): Initial horizontal angle for takeoff.
            initial_phi (float, optional): Initial vertical angle for takeoff.
        
        Returns:
            bool: True if takeoff successful, False otherwise.
        """
        if uav_id not in self._UAV_infos or not self._UAV_infos[uav_id]["is_docked"]:
            return False
        
        # Set initial values if provided, otherwise use defaults
        if initial_speed is None:
            initial_speed = random.uniform(self._UAV_speed_range[0], self._UAV_speed_range[1])
        if initial_angle is None:
            initial_angle = random.uniform(0, 2 * np.pi)
        if initial_phi is None:
            initial_phi = random.uniform(0, np.pi/6)  # Slightly upward
        
        # Update UAV info for takeoff
        self._UAV_infos[uav_id]["is_docked"] = False
        self._UAV_infos[uav_id]["docked_vehicle_id"] = None
        self._UAV_infos[uav_id]["speed"] = initial_speed
        self._UAV_infos[uav_id]["angle"] = initial_angle
        self._UAV_infos[uav_id]["phi"] = initial_phi
        
        # Update docking info
        if uav_id in self._UAV_docking_info:
            self._UAV_docking_info[uav_id]["is_docked"] = False
            self._UAV_docking_info[uav_id]["vehicle_id"] = None
        
        return True
    
    def setUAVDestination(self, uav_id, destination):
        """Set the destination for a UAV to fly to.
        
        Args:
            uav_id (str): The ID of the UAV.
            destination (tuple): The destination coordinates (x, y, z).
        
        Returns:
            bool: True if destination set successful, False otherwise.
        """
        if uav_id not in self._UAV_infos:
            return False
        
        # If UAV is hovering, can't set destination until resumed
        if self._UAV_infos[uav_id].get("is_hovering", False):
            print(f"Warning: UAV {uav_id} is hovering. Resume UAV first before setting destination.")
            return False
            
        self._UAV_infos[uav_id]["destination"] = destination
        return True
        
    def hoverUAV(self, uav_id):
        """Make a UAV hover at its current position.
        
        Args:
            uav_id (str): The ID of the UAV to hover.
            
        Returns:
            bool: True if hover command successful, False otherwise.
        """
        if uav_id not in self._UAV_infos:
            print(f"Warning: UAV {uav_id} does not exist")
            return False
            
        # Check if UAV is docked
        if self._UAV_infos[uav_id].get("is_docked", False):
            print(f"Warning: UAV {uav_id} is docked to a vehicle. Take off first before hovering.")
            return False
            
        # Check if UAV is already hovering
        if self._UAV_infos[uav_id].get("is_hovering", False):
            print(f"Note: UAV {uav_id} is already hovering.")
            return True
            
        # Save current speed and destination for later resuming
        uav_info = self._UAV_infos[uav_id]
        uav_info["hover_speed_before"] = uav_info.get("speed", 0)
        uav_info["hover_destination_before"] = uav_info.get("destination")
        
        # Set hovering state
        uav_info["is_hovering"] = True
        uav_info["speed"] = 0
        uav_info["acceleration"] = 0
        
        print(f"UAV {uav_id} is now hovering at position {uav_info['position']}")
        return True
        
    def resumeUAV(self, uav_id, resume_speed=None, resume_destination=True):
        """Resume a hovering UAV's movement.
        
        Args:
            uav_id (str): The ID of the UAV to resume.
            resume_speed (float, optional): Speed to resume with. If None, uses the speed before hovering.
            resume_destination (bool, optional): Whether to resume the previous destination. Defaults to True.
            
        Returns:
            bool: True if resume command successful, False otherwise.
        """
        if uav_id not in self._UAV_infos:
            print(f"Warning: UAV {uav_id} does not exist")
            return False
            
        # Check if UAV is hovering
        if not self._UAV_infos[uav_id].get("is_hovering", False):
            print(f"Warning: UAV {uav_id} is not in hovering state.")
            return False
            
        uav_info = self._UAV_infos[uav_id]
        
        # Determine speed to resume with
        if resume_speed is not None:
            new_speed = resume_speed
        elif uav_info["hover_speed_before"] is not None:
            new_speed = uav_info["hover_speed_before"]
        else:
            new_speed = random.uniform(self._UAV_speed_range[0], self._UAV_speed_range[1])
            
        # Set speed
        uav_info["speed"] = new_speed
        
        # Restore destination if requested
        if resume_destination and uav_info["hover_destination_before"] is not None:
            uav_info["destination"] = uav_info["hover_destination_before"]
            
        # Clear hovering state
        uav_info["is_hovering"] = False
        uav_info["hover_speed_before"] = None
        uav_info["hover_destination_before"] = None
        
        print(f"UAV {uav_id} has resumed movement with speed {new_speed}")
        return True
        
    def isUAVHovering(self, uav_id):
        """Check if a UAV is currently hovering.
        
        Args:
            uav_id (str): The ID of the UAV.
            
        Returns:
            bool: True if UAV is hovering, False otherwise.
        """
        if uav_id not in self._UAV_infos:
            return False
            
        return self._UAV_infos[uav_id].get("is_hovering", False)

    def stopVehicle(self, vehicle_id, duration, parking=True, stop_offset=20.0):
        """Make a vehicle stop at its current position for the specified duration.

        Args:
            vehicle_id (str): The ID of the vehicle to stop.
            duration (float): Duration of the stop in seconds.
            parking (bool, optional): Whether the vehicle is considered as parking 
                                    during the stop. Parking vehicles will not be
                                    interfered with by impatient drivers. Defaults to True.
            stop_offset (float, optional): Additional distance in meters to add to the current
                                        position for safer stopping. Defaults to 20.0.

        Returns:
            bool: True if the stop command was executed successfully, False otherwise.
        """
        if self._traffic_mode != 'SUMO':
            print("Warning: stopVehicle only works in SUMO mode")
            return False
            
        try:
            # Check if vehicle exists
            if vehicle_id not in self._traci_connection.vehicle.getIDList():
                print(f"Warning: Vehicle {vehicle_id} does not exist")
                return False
                
            # Get current lane information
            lane_id = self._traci_connection.vehicle.getLaneID(vehicle_id)
            if not lane_id:
                print(f"Warning: Vehicle {vehicle_id} is not on a lane")
                return False
                
            # Get current edge information
            edge_id = self._traci_connection.lane.getEdgeID(lane_id)
            
            # Get vehicle's position along the lane
            lane_pos = self._traci_connection.vehicle.getLanePosition(vehicle_id)
            
            # Add offset to lane position for safer stopping
            lane_length = self._traci_connection.lane.getLength(lane_id)
            # Make sure we don't exceed the lane length
            stop_position = min(lane_pos + stop_offset, lane_length - 1.0)
            
            # Get lane index within the edge
            lane_index = int(lane_id.split('_')[-1])
            
            # Set stop flags (1 for parking)
            stop_flags = 1 if parking else 0
            
            # Issue the stop command with the offset position
            self._traci_connection.vehicle.setStop(
                vehicle_id,     # Vehicle ID
                edge_id,        # Edge ID
                stop_position,  # Position along the edge with offset
                lane_index,     # Lane index
                duration,       # Duration
                stop_flags      # Flags (1 for parking)
            )
            
            return True
            
        except Exception as e:
            print(f"Error stopping vehicle {vehicle_id}: {str(e)}")
            return False

    def resumeVehicle(self, vehicle_id):
        """Cancels a previously issued stop command for the specified vehicle.
        
        Args:
            vehicle_id (str): The ID of the vehicle to resume movement.
            
        Returns:
            bool: True if the resume command was executed successfully, False otherwise.
        """
        if self._traffic_mode != 'SUMO':
            print("Warning: resumeVehicle only works in SUMO mode")
            return False
            
        try:
            # Check if vehicle exists
            if vehicle_id not in self._traci_connection.vehicle.getIDList():
                print(f"Warning: Vehicle {vehicle_id} does not exist")
                return False
                
            # Resume vehicle by canceling its stop with duration 0
            self._traci_connection.vehicle.resume(vehicle_id)
            return True
            
        except Exception as e:
            print(f"Error resuming vehicle {vehicle_id}: {str(e)}")
            return False

    def stopNearestVehicle(self, position, duration, max_distance=100.0, parking=True, stop_offset=20.0):
        """Find the nearest vehicle and stop it.
        
        Args:
            position (tuple): target positoin (x, y, z)
            duration (float): stop duration (seconds)
            max_distance (float, optional): search max distance. Defaults to 100.0.
            parking (bool, optional): whether to set as parking mode. Defaults to True.
            stop_offset (float, optional): Additional distance in meters to add to the current
                                        position for safer stopping. Defaults to 20.0.
            
        Returns:
            tuple: (success, vehicle_id or error message)
        """
        if self._traffic_mode != 'SUMO':
            return False, "only works in SUMO mode"
        
        # get all vehicles
        vehicle_ids = self._traci_connection.vehicle.getIDList()
        if not vehicle_ids:
            return False, "no available vehicles"
        
        # calculate the distance between each vehicle and the target position
        nearest_vehicle = None
        min_distance = float('inf')
        
        for vehicle_id in vehicle_ids:
            vehicle_pos = self._traci_connection.vehicle.getPosition(vehicle_id)
            # calculate 2D distance (ignore z-axis)
            distance = np.sqrt((vehicle_pos[0] - position[0])**2 + 
                              (vehicle_pos[1] - position[1])**2)
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_vehicle = vehicle_id
        
        if nearest_vehicle is None:
            return False, f"no vehicle found within {max_distance} meters"
        
        # stop the nearest vehicle
        success = self.stopVehicle(nearest_vehicle, duration, parking, stop_offset)
        if success:
            return True, nearest_vehicle
        else:
            return False, f"failed to stop vehicle {nearest_vehicle}"

    def _consumeUAVEnergy(self, UAV_id):
        """Consume energy for a flying UAV.
        
        Args:
            UAV_id (str): The ID of the UAV.
            
        Returns:
            bool: True if UAV has enough energy to continue flying, False otherwise.
        """
        if UAV_id not in self._UAV_infos:
            return False
        
        UAV_info = self._UAV_infos[UAV_id]
        
        # If UAV is docked, it doesn't consume energy
        if UAV_info.get("is_docked", False):
            return True
        
        # Get energy consumption rate
        consumption_rate = UAV_info.get("energy_consumption_rate", self._UAV_energy_consumption_rate)
        
        # Consume energy
        current_energy = UAV_info.get("current_energy", 0)
        if current_energy >= consumption_rate:
            UAV_info["current_energy"] = current_energy - consumption_rate
            return True
        else:
            UAV_info["current_energy"] = 0
            return False
    
    def getBatteryLevel(self, UAV_id):
        """Get the current battery level of a UAV as a percentage.
        
        Args:
            UAV_id (str): The ID of the UAV.
            
        Returns:
            float: The battery level as a percentage (0-100), or None if UAV not found.
        """
        if UAV_id not in self._UAV_infos:
            return None
            
        UAV_info = self._UAV_infos[UAV_id]
        battery_capacity = UAV_info.get("battery_capacity", self._UAV_battery_capacity)
        current_energy = UAV_info.get("current_energy", 0)
        
        if battery_capacity <= 0:
            return 0
            
        return (current_energy / battery_capacity) * 100