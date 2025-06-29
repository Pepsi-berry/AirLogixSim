from .abstract.simple_node import SimpleNode
import numpy as np


class UAV(SimpleNode):
    """The class for UAV.
    """
    def __init__(self, id, position, speed=0, acceleration=0, angle=0, phi=0, battery_capacity=100, energy_consumption_rate=1):
        """The constructor of the UAV class.
        Args:
            id (str): The unique ID of the UAV.
            position (tuple): The position of the UAV.
            speed (float): The speed of the UAV.
            acceleration (float): The acceleration of the UAV.
            angle (float): The angle of the UAV. The angle is the horizontal angle of the UAV.
            phi (float): The angle of the UAV. The phi is the vertical angle of the UAV.
            battery_capacity (float): The battery capacity of the UAV.
            energy_consumption_rate (float): The energy consumption rate per simulation step when flying.
        """
        position_x, position_y, position_z = position
        SimpleNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, angle)
        self._phi = phi
        self._last_updated_time = 0
        self._node_type = 'U'
        
        # Battery related attributes
        self._battery_capacity = battery_capacity
        self._current_energy = battery_capacity  # Start with full battery
        self._energy_consumption_rate = energy_consumption_rate
        
        # Docking related attributes
        self._is_docked = False
        self._docked_vehicle_id = None
        self._destination = None  # Target position to fly to when not docked
        
    def update(self, uav_traffic_info, simulation_time):
        """Update the UAV.

        Args:
            uav_traffic_info (dict): The traffic information of the UAV.
            simulation_time (float): The simulation time.
        """
        self._last_updated_time = simulation_time
        self._last_position = (self._position_x, self._position_y, self._position_z)
        self._position_x, self._position_y, self._position_z = uav_traffic_info['position']
        self._speed = uav_traffic_info['speed']
        self._acceleration = uav_traffic_info['acceleration']
        self._angle = uav_traffic_info['angle']
        self._phi = uav_traffic_info['phi']
        
        # Update battery status if provided in traffic info
        if 'current_energy' in uav_traffic_info:
            self._current_energy = uav_traffic_info['current_energy']
            
        # Update docking status if provided in traffic info
        if 'is_docked' in uav_traffic_info:
            self._is_docked = uav_traffic_info['is_docked']
        if 'docked_vehicle_id' in uav_traffic_info:
            self._docked_vehicle_id = uav_traffic_info['docked_vehicle_id']
        if 'destination' in uav_traffic_info:
            self._destination = uav_traffic_info['destination']

    def isMoving(self):
        return self._speed > 0

    def isDocked(self):
        """Check if UAV is currently docked to a vehicle.
        
        Returns:
            bool: True if UAV is docked, False otherwise.
        """
        return self._is_docked
    
    def getDockedVehicleId(self):
        """Get the ID of the vehicle the UAV is docked to.
        
        Returns:
            str: Vehicle ID if docked, None otherwise.
        """
        if self._is_docked:
            return self._docked_vehicle_id
        return None
    
    def dock(self, vehicle_id):
        """Dock the UAV to a vehicle.
        
        Args:
            vehicle_id (str): The ID of the vehicle to dock to.
        """
        self._is_docked = True
        self._docked_vehicle_id = vehicle_id
        self._speed = 0
        self._acceleration = 0
        # Recharge battery when docking
        self._current_energy = self._battery_capacity
        # Position will be updated in the traffic manager based on vehicle position
    
    def takeOff(self, initial_speed=20, initial_angle=0, initial_phi=0):
        """Take off the UAV from the docked vehicle.
        
        Args:
            initial_speed (float): Initial speed when taking off.
            initial_angle (float): Initial horizontal angle when taking off.
            initial_phi (float): Initial vertical angle when taking off.
        """
        self._is_docked = False
        self._speed = initial_speed
        self._angle = initial_angle
        self._phi = initial_phi
        # The UAV will maintain its current position at takeoff and then start moving
    
    def setDestination(self, destination):
        """Set the destination for the UAV to fly to.
        
        Args:
            destination (tuple): The destination coordinates (x, y, z).
        """
        self._destination = destination
    
    def getDestination(self):
        """Get the current destination of the UAV.
        
        Returns:
            tuple: The destination coordinates, or None if no destination is set.
        """
        return self._destination
    
    
    def consumeEnergy(self, amount=None):
        """Consume energy from the battery.
        
        Args:
            amount (float, optional): The amount of energy to consume. 
                If None, uses the default energy consumption rate.
                
        Returns:
            bool: True if enough energy is available, False if battery is depleted.
        """
        if amount is None:
            amount = self._energy_consumption_rate
            
        if self._current_energy >= amount:
            self._current_energy -= amount
            return True
        else:
            self._current_energy = 0
            return False
    
    def recharge(self, amount=None):
        """Recharge the UAV battery.
        
        Args:
            amount (float, optional): The amount of energy to add. 
                If None, fully recharges the battery.
        """
        if amount is None:
            self._current_energy = self._battery_capacity
        else:
            self._current_energy = min(self._current_energy + amount, self._battery_capacity)

    def to_dict(self):
        """Convert the UAV to a dictionary.

        Returns:
            dict: The UAV in dictionary format.
        """
        infos = {}
        uav_dict = SimpleNode.to_dict(self)
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                infos[key[1:]] = value
            else:
                infos[key] = value
        uav_dict.update(infos)
        return uav_dict