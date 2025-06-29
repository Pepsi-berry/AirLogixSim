from ..utils.enums import PackageState

class Task:
    def __init__(self, task_id, destination, package_weight, latest_delivery_time):
        """Initialize a delivery task.
        
        Args:
            task_id (str): Unique identifier for the task
            destination (tuple): Destination coordinates (x, y, z)
            package_weight (float): Weight of the package in kg
            latest_delivery_time (float): Latest time by which the package should be delivered
        """
        self._id = task_id
        self._destination = destination
        self._package_weight = package_weight
        self._latest_delivery_time = latest_delivery_time
        self._status = PackageState.WAITING
        
    def to_dict(self):
        """Convert task to dictionary representation."""
        infos = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                infos[key[1:]] = value
            else:
                infos[key] = value
        return infos
    
    def getId(self):
        """Get task ID."""
        return self._id
    
    def getDestination(self):
        """Get destination coordinates."""
        return self._destination
    
    def getPackageWeight(self):
        """Get package weight."""
        return self._package_weight
    
    def getLatestDeliveryTime(self):
        """Get latest delivery time."""
        return self._latest_delivery_time
    
    def getStatus(self):
        """Get task status."""
        return self._status
    
    def setStatus(self, status):
        """Set task status.
        
        Args:
            status (str): New status (pending, assigned, in_progress, completed, failed)
        """
        self._status = status
    
    def __str__(self):
        """String representation of the task."""
        return f"Task: id={self._id}, destination={self._destination}, package_weight={self._package_weight}kg, latest_delivery_time={self._latest_delivery_time}, status={self._status}" 