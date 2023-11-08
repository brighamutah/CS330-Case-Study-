import csv, json
import pandas as pd

drivers = pd.read_csv("drivers.csv")
passengers = pd.read_csv("passengers.csv")

# print(drivers.columns)
# print(passengers.columns)

with open('node_data.json', 'r') as f:
    node_data = json.load(f)

with open('adjacency.json', 'r') as f:
    adjacency = json.load(f)
# node_data_sorted = sorted(node_data, key=lambda x: x['lat'])

# n = len(node_data_sorted)


class Graph(object):
    def __init__(self, node_data_file, adjacency_file):
        self.graph = self.construct_graph(node_data_file, adjacency_file)

    def construct_graph(self, node_data_file, adjacency_file):
        # Read node data and adjacency list from JSON files
        with open(node_data_file, 'r') as file:
            node_data = json.load(file)
        
        with open(adjacency_file, 'r') as file:
            adjacency_list = json.load(file)

        # Initialize the graph with node data, including coordinates
        graph = {node_id: {'coords': {'lon': details['lon'], 'lat': details['lat']}, 'edges': {}}
                 for node_id, details in node_data.items()}

        # Update the graph with adjacency list data
        for start_node_id, edges in adjacency_list.items():
            for end_node_id, edge_attributes in edges.items():
                # Add the edge from start_node_id to end_node_id with its attributes
                graph[start_node_id]['edges'][end_node_id] = edge_attributes
                
                # Ensure the graph is symmetrical by adding the reverse edge with the same attributes
                if end_node_id not in graph:
                    graph[end_node_id] = {'coords': node_data[end_node_id], 'edges': {}}
                if start_node_id not in graph[end_node_id]['edges']:
                    graph[end_node_id]['edges'][start_node_id] = edge_attributes

        return graph

    def get_node_coordinates(self, node_id):
        # Returns the coordinates of the node
        return self.graph[node_id]['coords']

    def get_edge_attributes(self, start_node_id, end_node_id):
        # Returns the attributes of the edge between two nodes
        return self.graph[start_node_id]['edges'].get(end_node_id)
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node, time):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def find_nearest(lat, lon):
    # node_data_sorted is array of vertices (lat, lon) sorted by latitude
    print("fish")
    
# def lat_binary_search(t):
#     l = 0
#     r = n
#     while l <= r:
#         m = (l+r)/2
#         if t >= l[m]['lat']:
#             l = m
#         else:
#             r = m
    

def handle_passengers (nodes, passengers):
    while passengers:
        print(passengers[0])
        passengers.pop(0)
''''
Method to calculate time to destination for a passenger based off current time

'''
def dijkstra(nodes, nearest_node, current_time):
    unvisited_nodes = list(graph.get_nodes())
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path


# handle_passengers(1, [0,1,2,3])

graph = Graph("node_data.json", "adjacency.json")
print(graph)
    
    
    