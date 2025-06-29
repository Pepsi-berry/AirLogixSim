from graphviz import Digraph

def create_architecture_diagram():
    # Create a new directed graph
    dot = Digraph(comment='AirLogixSim Architecture')
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    # Set node styles
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Main environment cluster
    with dot.subgraph(name='cluster_env') as env:
        env.attr(label='Environment', style='rounded', color='blue', penwidth='2')
        
        # Core Environment
        env.node('env', 'AirLogixSimEnv\n(Core Environment)', fillcolor='lightgreen')
        
        # Middle Layer
        with env.subgraph(name='cluster_middle') as middle:
            middle.attr(label='Middle Layer', style='rounded')
            
            # Managers
            with middle.subgraph(name='cluster_managers') as managers:
                managers.attr(label='Managers', style='rounded')
                managers.node('traffic', 'TrafficManager\n(Traffic Control)')
                managers.node('task', 'TaskManager\n(Task Management)')
            
            # Visualization
            middle.node('visual', 'AirLogixSimEnvVisualizer\n(Visualization)')
        
        # Bottom Layer
        with env.subgraph(name='cluster_bottom') as bottom:
            bottom.attr(label='Bottom Layer', style='rounded')
            
            # Entities
            with bottom.subgraph(name='cluster_entities') as entities:
                entities.attr(label='Entities', style='rounded')
                entities.node('vehicle', 'Vehicle\n(Ground Vehicle)')
                entities.node('uav', 'UAV\n(Unmanned Aerial Vehicle)')
            
            # External Systems
            bottom.node('sumo', 'SUMO\n(Traffic Simulator)', fillcolor='lightyellow')
    
    # Utils (outside main environment)
    dot.node('utils', 'Utils\n(Helper Tools)', fillcolor='lightgray')
    
    # Algorithm (outside main environment)
    dot.node('algorithm', 'Algorithm\n(RL Agent)', fillcolor='lightpink')
    
    # Add edges
    # Core Environment connections
    dot.edge('env', 'traffic')
    dot.edge('env', 'task')
    dot.edge('env', 'visual')
    
    # Manager connections
    dot.edge('traffic', 'vehicle')
    dot.edge('traffic', 'uav')
    dot.edge('traffic', 'sumo')
    dot.edge('task', 'vehicle')
    dot.edge('task', 'uav')
    
    # Entity connections
    dot.edge('vehicle', 'uav', 'docks to')
    
    # Utils connections
    dot.edge('utils', 'visual', 'supports')
    dot.edge('utils', 'traffic', 'supports')
    dot.edge('utils', 'task', 'supports')
    
    # Algorithm interactions (RL loop)
    dot.edge('env', 'algorithm', 'state, reward')
    dot.edge('algorithm', 'env', 'action')
    
    # Save the diagram
    dot.render('airlogixsim_architecture', format='png', cleanup=True)

if __name__ == '__main__':
    create_architecture_diagram() 