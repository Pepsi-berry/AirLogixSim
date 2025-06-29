import os
import sys
import random
import numpy as np
import traci
import time

# Add the SUMO_HOME to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class SUMOSimulation:
    def __init__(self, sumocfg_file, fixed_destination=None, simulation_end_time=3600):
        self.sumocfg_file = sumocfg_file
        # 设置固定目的地，如果没有指定则为None
        self.fixed_destination = fixed_destination
        # 设置模拟结束时间（默认3600秒，即1小时）
        self.simulation_end_time = simulation_end_time
        self.traci_connection = self.init_traci()
        self.vehicle_id = "test_vehicle"
        # 获取有效边列表(供路径规划使用)
        self.valid_edges = self.get_valid_edges()
        # 计算间隔
        self.step_length = self.get_step_length()
        
    def get_step_length(self):
        """获取模拟步长"""
        try:
            return self.traci_connection.simulation.getDeltaT()
        except:
            return 0.1  # 默认步长为0.1秒
        
    def init_traci(self):
        """Initialize the TraCI connection to SUMO using a unique label"""
        sumo_label = "simulation_" + str(time.time())
        
        # Command list similar to airlogixsim_env.py
        cmd_list = ["sumo-gui", 
                   "--no-step-log", 
                   "--no-warnings", 
                   "--log", "sumo.log", 
                   "-c", self.sumocfg_file,
                   "--end", str(self.simulation_end_time)]  # 设置模拟结束时间
        
        print(f"Attempting to connect to SUMO with config file: {self.sumocfg_file}")
        print(f"设置模拟时长为 {self.simulation_end_time} 秒")
        
        # Start SUMO with a specific port and label
        try:
            traci.start(cmd_list, port=8813, label=sumo_label)
            
            # Get the connection with the specific label
            traci_connection = traci.getConnection(sumo_label)
            
            print(f"Successfully connected to SUMO with label: {sumo_label}")
            return traci_connection
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise
    
    def get_valid_edges(self):
        """获取网络中的有效边列表"""
        try:
            all_edges = self.traci_connection.edge.getIDList()
            # 过滤掉内部边和特殊边
            valid_edges = [e for e in all_edges if not e.startswith(":") and not e.startswith("-")]
            print(f"找到 {len(valid_edges)} 条有效边")
            return valid_edges
        except Exception as e:
            print(f"获取有效边列表时出错: {e}")
            return []
    
    def get_network_boundaries(self):
        """Get the boundaries of the SUMO network"""
        boundaries = self.traci_connection.simulation.getNetBoundary()
        return boundaries
    
    def add_vehicle(self):
        """添加一辆车到模拟中"""
        try:
            if not self.valid_edges:
                raise Exception("没有找到有效的边")
            
            # 选择一个有效的起始边
            start_edge = self.valid_edges[0]  # 使用第一个有效边
            
            # 创建一个路由ID并添加路由
            route_id = f"route_{self.vehicle_id}"
            # 添加一个简单路由，仅包含起始边
            self.traci_connection.route.add(route_id, [start_edge])
            
            # 创建车辆并分配路由
            print(f"添加车辆 {self.vehicle_id} 到边 {start_edge}")
            self.traci_connection.vehicle.add(
                vehID=self.vehicle_id,
                routeID=route_id
            )
            
            return start_edge
        except Exception as e:
            print(f"添加车辆时发生错误: {e}")
            raise
    
    def get_vehicle_edge(self):
        """获取车辆当前所在的边"""
        try:
            edge_id = self.traci_connection.vehicle.getRoadID(self.vehicle_id)
            print(f"车辆 {self.vehicle_id} 当前位于边 {edge_id}")
            return edge_id
        except Exception as e:
            print(f"获取车辆边时发生错误: {e}")
            raise
    
    def generate_destination(self, boundaries):
        """根据固定目的地或生成随机目的地"""
        if self.fixed_destination:
            print(f"使用固定目的地: {self.fixed_destination}")
            return self.fixed_destination
            
        # 否则生成随机目的地
        x_min, y_min = boundaries[0]
        x_max, y_max = boundaries[1]
        
        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)
        
        return (random_x, random_y)
    
    def find_nearest_edge(self, position):
        """找到离给定位置最近的边"""
        result = self.traci_connection.simulation.convertRoad(position[0], position[1])
        print(f"离位置 {position} 最近的边是: {result}")
        
        # 确保返回的边是有效的
        edge_id = result[0]  # 获取edge_id部分
        
        # 如果获取的边不在有效边列表中，尝试找一个有效的替代边
        if edge_id not in self.valid_edges and edge_id.lstrip('-') not in self.valid_edges:
            print(f"警告: 找到的边 {edge_id} 不在有效边列表中，尝试寻找替代边")
            
            # 获取目的地附近的车道位置
            lanes = self.traci_connection.lane.getIDList()
            closest_lane = None
            min_dist = float('inf')
            
            for lane in lanes:
                if lane.startswith(":"):  # 跳过内部车道
                    continue
                    
                try:
                    lane_shape = self.traci_connection.lane.getShape(lane)
                    # 计算车道中点到目的地的距离
                    mid_point = lane_shape[len(lane_shape)//2]
                    dist = ((mid_point[0] - position[0])**2 + (mid_point[1] - position[1])**2)**0.5
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_lane = lane
                except:
                    continue
            
            if closest_lane:
                edge_id = closest_lane.split("_")[0]  # 从车道ID提取边ID
                print(f"找到替代边: {edge_id}")
                
                # 确保是有效边
                if edge_id.startswith("-"):
                    edge_id = edge_id[1:]  # 移除负号
                
                # 最后检查是否在有效边列表中
                if edge_id in self.valid_edges:
                    return edge_id
            
            # 如果以上都失败，使用第一个有效边作为目标
            if self.valid_edges:
                print(f"无法找到合适的目标边，使用默认边: {self.valid_edges[0]}")
                return self.valid_edges[0]
        
        # 如果带负号，尝试使用不带负号的版本
        if edge_id.startswith("-") and edge_id[1:] in self.valid_edges:
            edge_id = edge_id[1:]
            
        return edge_id
    
    def plan_route(self, from_edge, to_edge):
        """规划从起始边到目标边的路径"""
        try:
            # 确保起始和目标边都是有效的
            if from_edge not in self.traci_connection.edge.getIDList():
                print(f"警告: 起始边 {from_edge} 不在网络中")
                if from_edge.startswith("-") and from_edge[1:] in self.valid_edges:
                    from_edge = from_edge[1:]
                    print(f"尝试使用替代起始边: {from_edge}")
                else:
                    from_edge = self.valid_edges[0]
                    print(f"使用默认起始边: {from_edge}")
            
            if to_edge not in self.traci_connection.edge.getIDList():
                print(f"警告: 目标边 {to_edge} 不在网络中")
                to_edge = self.valid_edges[-1]
                print(f"使用默认目标边: {to_edge}")
            
            # 尝试多次寻找有效路径
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    route = self.traci_connection.simulation.findRoute(from_edge, to_edge)
                    
                    # 检查是否找到了有效路径
                    if len(route.edges) > 0:
                        print(f"规划路径从 {from_edge} 到 {to_edge}")
                        print(f"路径长度: {route.length:.2f} 米, 花费: {route.cost:.2f}")
                        print(f"找到有效路径，包含 {len(route.edges)} 条边:")
                        for edge in route.edges:
                            print(f" - {edge}")
                        return route.edges
                    else:
                        print(f"第 {attempt+1} 次尝试: 未找到从 {from_edge} 到 {to_edge} 的路径")
                        
                        # 如果找不到路径，随机选择新的目标边
                        if attempt < max_attempts - 1:  # 不是最后一次尝试
                            to_edge = random.choice(self.valid_edges)
                            print(f"尝试新的目标边: {to_edge}")
                except traci.exceptions.TraCIException as e:
                    print(f"第 {attempt+1} 次尝试时出错: {e}")
                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        # 随机选择新边
                        from_edge, to_edge = random.sample(self.valid_edges, 2)
                        print(f"尝试新的边对: {from_edge} -> {to_edge}")
            
            print("无法找到有效路径，使用简单路径")
            # 如果所有尝试都失败，返回一个包含单个边的简单路径
            return [self.valid_edges[0]]
        except Exception as e:
            print(f"规划路径时发生错误: {e}")
            # 发生错误时返回一个简单路径
            return [self.valid_edges[0]] if self.valid_edges else []
    
    def set_vehicle_route(self, route_edges):
        """设置车辆的路径"""
        try:
            if not route_edges:
                print("警告: 路径为空，无法设置车辆路径")
                return False
                
            # 设置车辆路径
            print(f"正在设置车辆 {self.vehicle_id} 的路径，包含 {len(route_edges)} 条边")
            self.traci_connection.vehicle.setRoute(self.vehicle_id, route_edges)
            print(f"已成功设置车辆 {self.vehicle_id} 的路径")
            return True
        except Exception as e:
            print(f"设置车辆路径时发生错误: {e}")
            return False
    
    def run(self):
        """运行模拟"""
        try:
            # 获取网络边界
            boundaries = self.get_network_boundaries()
            print(f"网络边界: {boundaries}")
            
            # 步骤1: 添加车辆到模拟中
            start_edge = self.add_vehicle()
            
            # 执行一步模拟使车辆进入网络
            self.traci_connection.simulationStep()
            
            # 获取车辆当前位置（可能与初始放置位置不同）
            current_edge = self.get_vehicle_edge()
            
            # 步骤2: 生成目的地并找到最近的边
            destination_pos = self.generate_destination(boundaries)
            destination_edge = self.find_nearest_edge(destination_pos)
            
            # 步骤3: 规划路径
            route_edges = self.plan_route(current_edge, destination_edge)
            
            # 步骤4: 设置车辆路径
            route_set = self.set_vehicle_route(route_edges)
            
            if route_set:
                print("成功设置车辆路径，开始模拟")
                
                # 步骤5: 模拟车辆运行
                # 计算真实时间对应的模拟步数 (考虑到步长可能不是1秒)
                target_simulation_time = 1000  # 运行1000秒
                steps_per_second = 1.0 / self.step_length
                total_steps = int(target_simulation_time * steps_per_second)
                
                print(f"模拟步长为 {self.step_length} 秒，将运行 {total_steps} 步，约 {target_simulation_time} 秒")
                
                for step in range(total_steps):
                    try:
                        current_time = self.traci_connection.simulation.getTime()
                        self.traci_connection.simulationStep()
                        
                        # 每100步或每10秒打印一次车辆信息
                        if step % int(10 * steps_per_second) == 0:
                            try:
                                pos = self.traci_connection.vehicle.getPosition(self.vehicle_id)
                                speed = self.traci_connection.vehicle.getSpeed(self.vehicle_id)
                                edge = self.traci_connection.vehicle.getRoadID(self.vehicle_id)
                                print(f"步骤 {step}，模拟时间 {current_time:.1f}s: 车辆位置 {pos}, 速度 {speed:.2f}, 边 {edge}")
                            except traci.exceptions.TraCIException as e:
                                print(f"步骤 {step}，模拟时间 {current_time:.1f}s: 车辆已离开模拟 ({e})")
                                if "not known" in str(e):
                                    # 车辆已消失，可能需要重新添加
                                    print("尝试重新添加车辆...")
                                    try:
                                        self.add_vehicle()
                                        self.set_vehicle_route(route_edges)
                                    except:
                                        print("重新添加车辆失败")
                                
                    except traci.exceptions.FatalTraCIError as e:
                        print(f"SUMO关闭连接: {e}")
                        break
            else:
                print("无法设置车辆路径，只运行几步")
                for i in range(5):
                    self.traci_connection.simulationStep()
            
        except Exception as e:
            print(f"运行模拟时发生错误: {e}")
        finally:
            try:
                traci.close()
                print("模拟结束，已关闭SUMO连接")
            except:
                print("关闭SUMO连接时出错")

if __name__ == "__main__":
    # 获取当前脚本目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # SUMO配置文件的路径
    sumocfg_file = os.path.join(current_dir, "simple.sumocfg")
    
    print(f"使用配置文件: {sumocfg_file}")
    
    # 设置固定目的地为(530, 510)
    fixed_destination = (530, 510)
    
    # 创建并运行模拟，设置模拟结束时间为3600秒（1小时）
    simulation = SUMOSimulation(sumocfg_file, fixed_destination, simulation_end_time=3600)
    simulation.run() 