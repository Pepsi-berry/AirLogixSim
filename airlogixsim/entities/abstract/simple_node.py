class SimpleNode:
    def __init__(self, id, position_x, position_y, position_z, speed=0, acceleration=0, angle=0):
        self._id = id
        self._position_x = position_x
        self._position_y = position_y
        self._position_z = position_z
        self._speed = speed
        self._angle = angle
        self._acceleration = acceleration
        self._ai_model_dict = {}

    def to_dict(self):
        infos = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                infos[key[1:]] = value
            else:
                infos[key] = value
        return infos

    def updateAIModel(self, model_name, model):
        self._ai_model_dict[model_name] = model

    def getAIModel(self, model_name):
        return self._ai_model_dict[model_name]

    def getId(self):
        return self._id
    
    def getPosition(self):
        return (self._position_x, self._position_y, self._position_z)
    
    def getPosition2D(self):
        return (self._position_x, self._position_y)
    
    def getSpeed(self):
        return self._speed
    
    def getAngle(self):
        return self._angle
    
    def getAcceleration(self):
        return self._acceleration
    
    def setPosition(self, position_x, position_y, position_z):
        self._position_x = position_x
        self._position_y = position_y
        self._position_z = position_z

    def setSpeed(self, speed):
        self._speed = speed

    def setAcceleration(self, acceleration):
        self._acceleration = acceleration

    def __str__(self):
        return "SimpleNode: id={}, position=({}, {}, {}), speed={}, acceleration={}".format(self._id, self._position_x, self._position_y, self._position_z, self._speed, self._acceleration)