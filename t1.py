import json
import pandas as pd
import math
import heapq
from datetime import datetime, timedelta
import time
import random
import csv

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_csv_to_dict(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def construct_graph(adjacency_list):
    graph = {}
    for start_node, edges in adjacency_list.items():
        graph[start_node] = {}
        for end_node, attributes in edges.items():
            graph[start_node][end_node] = attributes
    return graph


def find_nearest_node(lat1, lon1, node_data):
    nearest_node = None
    min_distance = float('inf')

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    for node_id, coords in node_data.items():
        lat2 = math.radians(coords['lat'])
        lon2 = math.radians(coords['lon'])

        # From https://www.geeksforgeeks.org/program-distance-two-points-earth/
        distance = 3963.0 * math.acos((math.sin(lat1) * math.sin(lat2)) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
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

def reinsert_driver(drivers, driver, available):
    prob = 0.01 # Probability that driver is added back to available drivers
    rand = random.random()
    # = bool(random.getrandbits(1))
    if rand > prob: return drivers
    index = -1
    times = drivers['Date/Time'].tolist()
    n = len(drivers)
    l, r = 0, n-1
    # using binary search 
    while l <= r:
        m = (l+r)//2
        mid_time = times[m]
        if mid_time == available:
            index = m
            break
        elif mid_time < available:
            l = m + 1
        else:
            r = m - 1
            index = m

    driver['Date/Time'] = available
    drivers = pd.concat([drivers.iloc[:l], pd.DataFrame([driver]), drivers.iloc[l:]]).reset_index(drop=True)
    return drivers

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


        available_time = current_time + timedelta(hours = (time_to_passenger + time_to_destination))
        # Binary search driver times for time_to_destination to find where to reinsert driver
        drivers = reinsert_driver(drivers, driver, available_time)

        # Append calculated times to the respective lists
        wait_times.append(wait_time)
        profit_times.append(profit_time)

    # Compute average wait and profit times
    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0

    return average_wait_time, average_profit_time
#%%
adjacency_list = load_json("adjacency.json")
node_data = load_json("node_data.json")
drivers_data = load_csv("drivers.csv")
passengers_data = load_csv("passengers.csv")

drivers_data['Date/Time'] = pd.to_datetime(drivers_data['Date/Time'])
passengers_data['Date/Time'] = pd.to_datetime(passengers_data['Date/Time'])

drivers_data.sort_values(by='Date/Time', inplace=True)
passengers_data.sort_values(by='Date/Time', inplace=True)

graph = construct_graph(adjacency_list)

#%%
start_time = time.time()
average_wait_time, average_profit_time = match_and_calculate_metrics(drivers_data, passengers_data, graph, node_data)
end_time = time.time()

print("Average Wait Time for Passengers (D1):", average_wait_time, "minutes")
print("Average Profit Time for Drivers (D2):", average_profit_time, "minutes")
print(f"Runtime (excluding loading data): {end_time-start_time}")

