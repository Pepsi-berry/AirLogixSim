import numpy as np
import random
import sumolib
from airlogixsim.entities.task import Task
from ..utils.enums import UAVState, VehicleState, PackageState,UAVActionRet

class TaskManager:
    """Task manager for delivery tasks. 
    Handles task initialization, generation, and management.
    """
    
    def __init__(self, config_task, traffic_manager=None, sumo_network_xml=None):
        """Initialize the task manager.
        
        Args:
            config_task (dict): Task configuration
            traffic_manager: Reference to the traffic manager (optional)
            sumo_network_xml (str): Path to SUMO network XML file
        """
        # Set random seed for reproducibility
        random.seed(42)  # 使用固定的种子值42
        np.random.seed(42)  # 同时也设置numpy的随机种子
        
        self._config_task = config_task
        self._traffic_manager = traffic_manager
        
        # Map boundaries (get from traffic manager if available, otherwise from config)
        if traffic_manager:
            self._x_range = traffic_manager._x_range
            self._y_range = traffic_manager._y_range
        else:
            self._x_range = config_task.get("x_range", [0, 3000])
            self._y_range = config_task.get("y_range", [0, 3000])
        
        # Load SUMO network
        self._net = None
        if sumo_network_xml:
            self._net = sumolib.net.readNet(sumo_network_xml)
        
        # Task settings
        self._min_weight = config_task.get("min_package_weight", 0.5)  # kg
        self._max_weight = config_task.get("max_package_weight", 5.0)  # kg
        self._min_delivery_time = config_task.get("min_delivery_time", 300)  # seconds
        self._max_delivery_time = config_task.get("max_delivery_time", 1800)  # seconds
        
        # Task storage
        self._tasks = {}  # task_id -> Task object
        self._task_id_counter = 0
        
    def reset(self):
        """Reset the task manager."""
        self._tasks = {}
        self._task_id_counter = 0
        
    def _generate_random_destination(self):
        """Generate random destination within SUMO map bounds.
        
        Returns:
            tuple: (x, y, z) coordinates
        """
        if self._net:
            # Get all junctions in the network
            junctions = self._net.getNodes()
            if junctions:
                # Select a random junction
                junction = random.choice(junctions)
                x, y = junction.getCoord()
                z = 0  # Destinations are ground-level
                return (x, y, z)
                
        # Fallback if SUMO net is not available or has no junctions
        x = random.uniform(self._x_range[0], self._x_range[1])
        y = random.uniform(self._y_range[0], self._y_range[1])
        z = 0  # Destinations are ground-level
        return (x, y, z)
    
    def _generate_random_weight(self):
        """Generate random package weight.
        
        Returns:
            float: Package weight in kg
        """
        return round(random.uniform(self._min_weight, self._max_weight), 2)
    
    def _generate_random_delivery_time(self, current_time=0):
        """Generate random latest delivery time.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            float: Latest delivery time
        """
        time_window = random.uniform(self._min_delivery_time, self._max_delivery_time)
        return current_time + time_window
    
    def create_task(self, destination=None, package_weight=None, latest_delivery_time=None, current_time=0):
        """Create a new task with specified or random parameters.
        
        Args:
            destination (tuple, optional): Task destination (x, y, z)
            package_weight (float, optional): Package weight in kg
            latest_delivery_time (float, optional): Latest delivery time
            current_time (float): Current simulation time
            
        Returns:
            Task: Created task object
        """
        task_id = f"task_{self._task_id_counter}"
        self._task_id_counter += 1
        
        # Use provided values or generate random ones

        # generate random destination
        # In some cases, the destination should be regenerated: 
        # 1. the destination is too close to other tasks
        
        if destination is None:
            while True:
                destination = self._generate_random_destination()
                if self._is_destination_valid(destination):
                    break

        
        if package_weight is None:
            package_weight = self._generate_random_weight()
            
        if latest_delivery_time is None:
            latest_delivery_time = self._generate_random_delivery_time(current_time)
        
        # Create task
        task = Task(task_id, destination, package_weight, latest_delivery_time)
        
        # Store task
        self._tasks[task_id] = task
        
        return task
    
    def _is_too_close(self, destination1, destination2):
        """Check if the destination is too close to other tasks.
        
        Args:
            destination1 (tuple): Destination coordinates (x, y, z)
            destination2 (tuple): Destination coordinates (x, y, z)
        """
        return np.linalg.norm(np.array(destination1) - np.array(destination2)) < 10
    
    def _is_destination_valid(self, destination):
        """Check if the destination is valid.

        Args:
            destination (tuple): Destination coordinates (x, y, z)
            
        Returns:
            bool: True if the destination is valid, False otherwise
        """
        # check if the destination is too close to other tasks
        # iterate through all tasks and check the distance
        for task in self._tasks.values():
            if self._is_too_close(destination, task.getDestination()):
                return False
        return True


    def initialize_tasks(self, num_tasks, current_time=0):
        """Initialize a batch of random tasks.
        
        Args:
            num_tasks (int): Number of tasks to initialize
            current_time (float): Current simulation time
            
        Returns:
            dict: Created tasks by task_id
        """
        ## refresh the task manager first
        self.reset()
        created_tasks = {}
        
        for _ in range(num_tasks):
            task = self.create_task(current_time=current_time)
            created_tasks[task.getId()] = task
            
        return created_tasks
    
    def get_task(self, task_id):
        """Get task by ID.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            Task: Task object or None if not found
        """
        return self._tasks.get(task_id)
    
    def get_task_locations_2d(self):
        """Get all task locations in 2D.
           return is in the form of numpy array
        """
        return np.array([task.getDestination()[:2] for task in self._tasks.values()])
    
    def get_task_weights(self):
        """Get all task weights.
           return is in the form of numpy array
        """
        return np.array([task.getPackageWeight() for task in self._tasks.values()])
    
    def get_all_tasks(self):
        """Get all tasks.
        
        Returns:
            dict: All tasks by task_id
        """
        return self._tasks
    
    def get_num_of_tasks(self):
        """Get the number of tasks.
        
        Returns:
            int: Number of tasks
        """
        return len(self._tasks)
    
    def get_tasks_by_status(self, status):
        """Get tasks with specified status.
        
        Args:
            status (str): Task status (pending, assigned, in_progress, completed, failed)
            
        Returns:
            dict: Tasks with specified status by task_id
        """
        filtered_tasks = {}
        for task_id, task in self._tasks.items():
            if task.getStatus() == status:
                filtered_tasks[task_id] = task
        return filtered_tasks
        
    @classmethod
    def from_config(cls, config, traffic_manager=None):
        """Create a TaskManager instance from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary containing task settings
            traffic_manager: Reference to the traffic manager (optional)
            
        Returns:
            TaskManager: Instance initialized with the provided configuration
        """
        # Extract task-specific configuration
        config_task = config.get("task", {})
        
        # Get SUMO network path from configuration
        sumo_net_path = config.get("sumo", {}).get("sumo_net", None)
        
        # Create and return TaskManager instance
        return cls(config_task, traffic_manager, sumo_net_path)
