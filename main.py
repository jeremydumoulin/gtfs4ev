# coding: utf-8

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import math

from gtfs4ev.vehicle import Vehicle
from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.vehicle import Vehicle
from gtfs4ev import helpers as hlp

"""
Main function
"""
def main():

	feed = GTFSFeed("GTFS_Nairobi")		
	feed.clean_trips()


	# trips = feed.trips['trip_id']

	# print(trips)

	# output_data = []
	# df = pd.DataFrame()
	# time_step = 100
	# time_span = 54000
	# tot_energy = .0

	# profile = np.zeros(int(time_span/time_step))

	# i = 0
	# for trip in trips:
	# 	i += 1		
	# 	print(f"Completion: {i / len(trips) * 100}%")
	# 	vehicle = Vehicle(feed = feed, trip_id=trip, ev_consumption = 0.4)

	# 	print(vehicle.trip_frequencies)

	# 	energy = vehicle.trip_frequencies['energy_estimate'].sum()
	# 	# print(vehicle.trip_frequencies['energy_estimate'])
	# 	# print(energy)
	# 	tot_energy += energy

	# 	vehicle.simulate(time_span, time_step)

	# 	# print(vehicle.trip_profile['power_kW'])

	# 	profile += vehicle.trip_profile['power_kW'].values
				
	# 	# vehicle.simulate(time_span, time_step)
	# 	# print(vehicle.trip_profile)

	# print(profile)
	# # 



	# vehicle = Vehicle(feed = feed, trip_id='20045131', ev_consumption = 0.2)
	# print(vehicle.trip_frequencies)
	# vehicle.simulate(54000, 50)


	# statistics = vehicle.statistics()	

	# print(statistics)
	# print(vehicle.trip_frequencies)
	# print(vehicle.trip_data.iloc[-1])

	# vehicle.simulate(54000, 50)

	# df = vehicle.trip_profile

	# # Calculate moving average for the 'Power' column
	# window_size = 120
	# df['Power_MA'] = df['power_kW'].rolling(window=window_size, min_periods=1).mean()

	# # Plotting
	# plt.figure(figsize=(10, 6))

	# # Plot power values
	# plt.plot(df['t'], df['power_kW'], label='Power', color='green', linestyle='-')

	# # Plot moving average line
	# plt.plot(df['t'], df['Power_MA'], color='black', label=f'Power Moving Avg (window={window_size})', linestyle='--')


	# # Plot energy values on a second y-axis
	# ax2 = plt.gca().twinx()
	# ax2.plot(df['t'], df['energy_kWh'], color='orange', label='Energy', linestyle='--')

	# # Set labels and title
	# plt.xlabel('Time')
	# plt.ylabel('Power (kW)')
	# ax2.set_ylabel('Cumulative Energy (kWh)')

	# plt.title('Power and Energy Over Time')

	# plt.show()



	time_values = np.arange(0, 4000, 20) 

	vehicle = Vehicle(feed = feed, trip_id='20237110', ev_consumption = 0.4)

	power_values = [vehicle.power_profile(t) for t in time_values]

	# Plot the function
	plt.plot(time_values, power_values, marker='o')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Power')
	plt.title('Power vs. Time')
	plt.grid(True)

	plt.savefig('power_vs_time.png')
	plt.show()




	# def total_power(headway, n_vehicles):
	# 	for i in range(0, n_vehicles):
	# 		print(i)

	# total_power(300, 12)

	# start_time = filtered_frequencies.iloc[0]['start_time']
	# stop_time = filtered_frequencies.iloc[0]['end_time']
	# time_step = 10

	# headway_time = 300 # headway time in seconds


	



	# print(vehicle.trip_data)
	# print(vehicle.power_profile(149))
	# print(vehicle.energy_profile(149))
	# print(vehicle.speed_profile(149))

	# print(vehicle.statistics())


	# # Generate a range of time values
	# time_values = np.arange(0, 3000, 5)  

	# # Apply the lambda function to get power values
	# power_values = [power_profile(t) for t in time_values]
	# energy_values = [energy_profile(t) for t in time_values]

	# # Plot the function
	# plt.plot(time_values, energy_values, marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# plt.savefig('power_vs_time.png')
	# plt.show()


	#feed.general_feed_info()

	# print(f"min_lat: {feed.min_latitude}")	
	# print(f"max_lat: {feed.max_latitude}")	
	# print(f"min_lon: {feed.min_longitude}")	
	# print(f"max_lon: {feed.max_longitude}")	

	# print(f"Bounding box: {feed.bounding_box}")	

	#print(feed.simulation_area())



	# trip_id = "1107D110"

	# # Get the shape corresponding to the trip

	# gdf = pd.merge(feed.trips, feed.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
	# linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

	# # Get the stops of the trip with the corresponding latitude and longitude
	
	# filtered_stop_times = feed.stop_times[feed.stop_times['trip_id'] == trip_id]

	# # Merge the filtered_df with the stop_points_df on 'stop_id'
	# result_df = pd.merge(filtered_stop_times, feed.stops[['stop_id', 'stop_name', 'geometry']], on='stop_id', how='left')

	# # Add the closest point on the path as the point for the stop
	# result_df['closest_point'] = result_df['geometry'].apply(lambda point: hlp.find_closest_point(linestring, point))
	
	# # Initialize an empty list to store the data
	# new_data = []

	# # Iterate over the DataFrame to get the departure and arrival times between two consecutive stops
	# # Skip the first row
	# for i in range(1, len(result_df)):
	#     current_stop = result_df.iloc[i]
	#     previous_stop = result_df.iloc[i - 1]

	#     trip_id = current_stop['trip_id']

	#     # Add the information for the current stop 
	#     new_data.append({
	#         'trip_id': trip_id,
	#         'moving': False,
	#         'departure_time': current_stop['departure_time'],
	#         'arrival_time': current_stop['arrival_time'],
	#         'start_stop_name': current_stop['stop_name'],
	#         'end_stop_name': current_stop['stop_name'],
	#         'start_stop_loc': current_stop['closest_point'],
	#         'end_stop_loc': current_stop['closest_point'],
	#         'duration': (current_stop['departure_time']-current_stop['arrival_time']).total_seconds(),
	#         'distance': .0,
	#         'average_speed': .0,
	#         'electrical_power': .0
	#     })
	    

	    # # Extract relevant information
	    
	    # departure_time = previous_stop['departure_time']
	    # arrival_time = current_stop['arrival_time']
	    # start_stop_name = previous_stop['stop_name']
	    # end_stop_name = current_stop['stop_name']
	    # start_stop_loc = previous_stop['closest_point']
	    # end_stop_loc = current_stop['closest_point']

	    # # Define the projection from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
	    # web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 
	    # line_shape_web_mercator = transform(web_mercator_projection, linestring)
	    # point1_web_mercator = transform(web_mercator_projection, start_stop_loc)
	    # point2_web_mercator = transform(web_mercator_projection, end_stop_loc)

	    # distance_along_line = abs(line_shape_web_mercator.project(point1_web_mercator) - line_shape_web_mercator.project(point2_web_mercator))

	    # ave_speed = distance_along_line / (arrival_time-departure_time).total_seconds() * 3.6 # Average speed in km/h

	    # # Append the data to the list
	    # new_data.append({
	    #     'trip_id': trip_id,
	    #     'departure_time': departure_time,
	    #     'arrival_time': arrival_time,
	    #     'start_stop_name': start_stop_name,
	    #     'end_stop_name': end_stop_name,
	    #     'start_stop_loc': start_stop_loc,
	    #     'end_stop_loc': end_stop_loc,
	    #     'distance': distance_along_line,
	    #     'average_speed': ave_speed
	    # })

	# Create a new DataFrame from the list
	# new_df = pd.DataFrame(new_data)

	# # Print the new DataFrame
	# print(new_df)
		

if __name__ == "__main__":
	main()