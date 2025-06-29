from graphviz import Digraph

def create_stop_vehicle_flowchart():
    # Create a new directed graph
    dot = Digraph(comment='StopVehicle Algorithm Flowchart')
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    # Set node styles
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Start node
    dot.node('start', 'Start', fillcolor='lightgreen')
    
    # Check SUMO mode
    dot.node('check_sumo', 'Check if in SUMO mode?', shape='diamond', fillcolor='lightyellow')
    dot.edge('start', 'check_sumo')
    
    # SUMO mode check branches
    dot.node('error_sumo', 'Return False\nPrint Warning', fillcolor='lightpink')
    dot.edge('check_sumo', 'error_sumo', 'No')
    
    # Try block
    dot.node('try_block', 'Try Block', fillcolor='lightgray')
    dot.edge('check_sumo', 'try_block', 'Yes')
    
    # Vehicle existence check
    dot.node('check_vehicle', 'Check if vehicle exists?', shape='diamond', fillcolor='lightyellow')
    dot.edge('try_block', 'check_vehicle')
    
    # Vehicle existence branches
    dot.node('error_vehicle', 'Return False\nPrint Warning', fillcolor='lightpink')
    dot.edge('check_vehicle', 'error_vehicle', 'No')
    
    # Lane check
    dot.node('check_lane', 'Get and check lane information', shape='diamond', fillcolor='lightyellow')
    dot.edge('check_vehicle', 'check_lane', 'Yes')
    
    # Lane check branches
    dot.node('error_lane', 'Return False\nPrint Warning', fillcolor='lightpink')
    dot.edge('check_lane', 'error_lane', 'Invalid')
    
    # Get edge information
    dot.node('get_edge', 'Get edge information', fillcolor='lightblue')
    dot.edge('check_lane', 'get_edge', 'Valid')
    
    # Get lane position
    dot.node('get_position', 'Get vehicle lane position', fillcolor='lightblue')
    dot.edge('get_edge', 'get_position')
    
    # Calculate stop position
    dot.node('calc_position', 'Calculate stop position\nwith safety offset', fillcolor='lightblue')
    dot.edge('get_position', 'calc_position')
    
    # Get lane index
    dot.node('get_index', 'Get lane index', fillcolor='lightblue')
    dot.edge('calc_position', 'get_index')
    
    # Set stop flags
    dot.node('set_flags', 'Set stop flags\n(parking mode)', fillcolor='lightblue')
    dot.edge('get_index', 'set_flags')
    
    # Issue stop command
    dot.node('issue_stop', 'Issue stop command\nwith all parameters', fillcolor='lightblue')
    dot.edge('set_flags', 'issue_stop')
    
    # Return success
    dot.node('success', 'Return True', fillcolor='lightgreen')
    dot.edge('issue_stop', 'success')
    
    # Exception handling
    dot.node('catch', 'Catch Exception', fillcolor='lightpink')
    dot.edge('try_block', 'catch')
    dot.edge('catch', 'error_vehicle')
    
    # End node
    dot.node('end', 'End', fillcolor='lightgreen')
    dot.edge('success', 'end')
    dot.edge('error_vehicle', 'end')
    dot.edge('error_lane', 'end')
    dot.edge('error_sumo', 'end')
    
    # Save the diagram
    dot.render('stop_vehicle_flowchart', format='png', cleanup=True)

if __name__ == '__main__':
    create_stop_vehicle_flowchart() 