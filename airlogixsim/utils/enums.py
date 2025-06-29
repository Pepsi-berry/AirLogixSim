# airlogixsim/utils/enums.py
from enum import Enum

class UAVState(Enum):
    IDLE = 0
    DELIVERING = 1
    LANDING = 2
    RETURNING = 3
    INIT = 4


class VehicleState(Enum):
    MOVING = 1
    LANDING = 0


class PackageState(Enum):
    WAITING = 0
    ASSIGNED = 1
    DELIVERED = 2


class UAVActionRet(Enum):
    FEASIBLE = 0  # uav can move
    OUT_OF_CLUSTER = 1  # node is out of current k-means cluster
    OUT_OF_POWER = 2  # uav will be out of power if it moves
    OUT_OF_CAPACITY = 3  # uav will be out of capacity if it moves
    CLOSED_NODE = 4  # node has been assigned or delivered
    CANNOT_RETURN = 5  # uav cannot return to the truck if it moves
    SAME_TARGET = 6  # uav choose truck repeatedly