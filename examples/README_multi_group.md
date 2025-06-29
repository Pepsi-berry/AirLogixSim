# 多组车辆无人机协同配送任务分配算法

## 概述

本算法是对原有单组车辆无人机配送系统的重大升级，支持多组车辆无人机协同工作，共同完成配送任务。

## 主要特性

### 1. 多组协同支持
- 支持多个车辆无人机组同时运行
- 每组包含多辆车和多架无人机
- 组间任务协调，避免冲突

### 2. 智能集群分配
- 基于K-means聚类的任务分区
- 动态集群分配策略
- 考虑距离、任务密度和紧急程度

### 3. 高级任务调度
- 无人机智能任务选择
- 电量和载重优化
- 距离和重量优先级排序

## 算法架构

```
MultiGroupTaskAllocator
├── 主要任务分配函数 (allocate_tasks)
├── 组任务分配 (_allocate_tasks_for_group)
├── 车辆任务分配 (_assign_vehicle_task)
├── 无人机任务分配 (_assign_uav_task)
└── 智能任务选择 (_smart_uav_task_selection)
```

## 配置参数

### 配置文件示例 (config.yaml)

```yaml
traffic:
  group_num: 3              # 组数量
  vehicle_count: 2          # 每组车辆数
  UAV_count: 3             # 每组无人机数
  
task:
  num_tasks: 30            # 任务总数

clustering:
  cluster_number: 5        # 集群数量
```

## 算法逻辑

### 1. 车辆任务分配策略
```python
def _assign_vehicle_task(self, vehicle_id, group_num, observation, info):
    """
    为车辆选择最优集群中心节点
    - 避免与其他组冲突
    - 考虑集群完成度
    - 评估距离和任务密度
    """
```

### 2. 无人机任务选择策略
```python
def _smart_uav_task_selection(self, uav_id, group_num, assigned_cluster, observation, info):
    """
    三层策略：
    1. 电量/载重检查 -> 返回车辆
    2. 集群内任务选择 -> 配送任务
    3. 备选策略 -> 返回车辆
    """
```

### 3. 集群评分机制
- **节点数量评分** (40%): 适中的任务密度
- **距离评分** (30%): 距离仓库的远近
- **紧急程度评分** (30%): 集群完成进度

## 使用方法

### 1. 基本使用
```python
# 创建环境
env = AirLogixSimEnv(config, interactive_mode=None)
observation, info = env.reset()

# 创建多组任务分配器
task_allocator = MultiGroupTaskAllocator(env)

# 主循环
while running:
    # 使用升级的任务分配算法
    action = task_allocator.allocate_tasks(observation, info)
    observation, reward, termination, truncation, info = env.step(action, training=True)
```

### 2. 运行仿真
```bash
cd examples
python run_simulation.py
```

## 性能优势

### 与单组算法对比
| 特性 | 单组算法 | 多组算法 |
|------|----------|----------|
| 并行处理 | ❌ | ✅ |
| 集群智能分配 | ❌ | ✅ |
| 组间协调 | ❌ | ✅ |
| 任务冲突避免 | ❌ | ✅ |
| 负载均衡 | ❌ | ✅ |

### 关键改进
1. **并行效率**: 多组同时工作，显著提高配送效率
2. **智能调度**: 基于距离、重量、电量的综合决策
3. **冲突避免**: 组间集群分配协调机制
4. **动态适应**: 根据完成情况动态调整策略

## 系统监控

算法提供详细的状态监控：

```python
# 无人机状态
print("无人机状态:")
for uav_id, status in uav_status.items():
    print(f"  {uav_id}: 载重:{status['capacity']}, 电量:{status['power']}")

# 组分配状态
print("组集群分配状态:")
for group_num, cluster_id in task_allocator.group_cluster_assignments.items():
    print(f"  组 {group_num}: 集群 {cluster_id}")
```

## 扩展性

### 支持的组合
- **小规模**: 2组 × (1车+2无人机) = 6个智能体
- **中规模**: 3组 × (2车+3无人机) = 15个智能体  
- **大规模**: 5组 × (3车+4无人机) = 35个智能体

### 自定义策略
可以通过继承`MultiGroupTaskAllocator`类来实现自定义的任务分配策略：

```python
class CustomTaskAllocator(MultiGroupTaskAllocator):
    def _calculate_task_priority(self, distance, weight, node_idx):
        # 自定义优先级计算
        return custom_score
```

## 注意事项

1. **配置一致性**: 确保`group_num`、`vehicle_count`、`UAV_count`与环境设置一致
2. **集群数量**: `cluster_number`应适当大于`group_num`以避免组间冲突
3. **计算复杂度**: 组数增加会提高计算复杂度，建议根据硬件性能调整

## 故障排除

### 常见问题
1. **组分配冲突**: 增加`cluster_number`
2. **无人机卡死**: 检查电量和载重设置
3. **收敛缓慢**: 调整任务优先级权重

### 调试建议
- 启用详细日志输出
- 监控组分配状态
- 观察集群完成进度

---

**开发团队**: AirLogixSim 开发组  
**版本**: 2.0  
**更新日期**: 2024年12月 