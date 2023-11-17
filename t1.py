import json
import pandas as pd
import math
import heapq
from datetime import datetime

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_csv(file_path):
    return pd.read_csv(file_path)


def construct_graph(adjacency_list):
    graph = {}
    for start_node, edges in adjacency_list.items():
        graph[start_node] = {}
        for end_node, attributes in edges.items():
            graph[start_node][end_node] = attributes
    return graph


def find_nearest_node(lat, lon, node_data):
    nearest_node = None
    min_distance = float('inf')

    for node_id, coords in node_data.items():
        distance = math.sqrt((coords['lat'] - lat) ** 2 + (coords['lon'] - lon) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node


def calculate_route_time(start_node, end_node, graph, current_time):
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    pq = [(0, start_node)]

    current_hour = current_time.hour
    day_type = 'weekday' if current_time.weekday() < 5 else 'weekend'

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            return distances[end_node]

        for neighbor, attributes_list in graph[current_node].items():
            for attributes in attributes_list:
                if attributes['day_type'] == day_type and attributes['hour'] == current_hour:
                    distance = current_distance + attributes['time']
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                    break

    return float('infinity')


def match_and_calculate_metrics(drivers, passengers, graph, node_data):
    wait_times = []
    profit_times = []

    while not drivers.empty and not passengers.empty:
        driver = drivers.iloc[0]
        passenger = passengers.iloc[0]

        # Remove the matched driver and passenger from their respective queues
        drivers = drivers.iloc[1:]
        passengers = passengers.iloc[1:]

        # Find the nearest nodes in the graph to the driver and passenger locations
        driver_node = find_nearest_node(driver['Source Lat'], driver['Source Lon'], node_data)
        passenger_pickup_node = find_nearest_node(passenger['Source Lat'], passenger['Source Lon'], node_data)
        passenger_dropoff_node = find_nearest_node(passenger['Dest Lat'], passenger['Dest Lon'], node_data)

        # Use the 'Date/Time' field from the driver's data as the current time
        current_time = driver['Date/Time']

        # Calculate the time to passenger and time to destination
        time_to_passenger = calculate_route_time(driver_node, passenger_pickup_node, graph, current_time)
        time_to_destination = calculate_route_time(passenger_pickup_node, passenger_dropoff_node, graph, current_time)

        # Calculate wait time for the passenger and profit time for the driver
        wait_time = time_to_passenger  # Time from driver's availability to passenger pickup
        profit_time = time_to_destination - time_to_passenger  # Time driving the passenger minus time to reach them

        # Append calculated times to the respective lists
        wait_times.append(wait_time)
        profit_times.append(profit_time)

    # Compute average wait and profit times
    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0

    return average_wait_time, average_profit_time



adjacency_list = load_json("adjacency.json")
node_data = load_json("node_data.json")
drivers_data = load_csv("drivers.csv")
passengers_data = load_csv("passengers.csv")

drivers_data['Date/Time'] = pd.to_datetime(drivers_data['Date/Time'])
passengers_data['Date/Time'] = pd.to_datetime(passengers_data['Date/Time'])


drivers_data.sort_values(by='Date/Time', inplace=True)
passengers_data.sort_values(by='Date/Time', inplace=True)

graph = construct_graph(adjacency_list)

average_wait_time, average_profit_time = match_and_calculate_metrics(drivers_data, passengers_data, graph, node_data)

print("Average Wait Time for Passengers (D1):", average_wait_time, "minutes")
print("Average Profit Time for Drivers (D2):", average_profit_time, "minutes")

