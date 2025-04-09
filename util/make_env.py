from env.MultiAgentEnv import DeliveryEnv

# 辅助函数：创建环境的构造函数
def make_env(config=None):
    def _init():
        return DeliveryEnv(config)
    return _init