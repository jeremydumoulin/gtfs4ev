# coding: utf-8

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import math
import folium
from folium.plugins import MarkerCluster, MeasureControl

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

"""
Main function
"""
def main():

	"""
	Setting up and cleaning the GTFS Feed
	"""

	# Populate the feed with the raw data
	feed = GTFSFeed("GTFS_Nairobi")	

	# Check if there is any issue with de data
	if feed.data_check():
		feed.clean_all()

	# Recheck the data to make sure everything has been cleaned
	feed.data_check()

	# If needed, filter out trips belonging to specific services
	#feed.filter_services('service_0001')

	"""
	1. Get general information about the feed
	"""

	feed.general_feed_info()

	"""
	2. Visualize the paratransit network
	"""

	# """ Visualize a trip using the trip_id """
	# trip_id = '1107D110'

	# # Get the shape of the trip_id
	# shape = feed.get_shape(trip_id) 

	# # Extract coordinates from the LineString
	# coordinates = list(shape.coords)

	# # Create a Folium Map centered at the midpoint of the LineString
	# midpoint = (shape.centroid.y, shape.centroid.x)
	# mymap = folium.Map(location=midpoint, zoom_start=12)

	# # Add the LineString to the map
	# folium.PolyLine(locations=[[coord[1], coord[0]] for coord in shape.coords], color='blue').add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/trip_{trip_id}.html")	

	# """ Add the stops to the path """
	# # Get the stop locations
	# stop_coordinates = feed.get_stop_locations(trip_id) 

	# # Add the Points to the map
	# for point in stop_coordinates:
	#     folium.Marker([point.y, point.x], icon=folium.Icon(color='red')).add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/trip_{trip_id}_withstops.html")

	# """ Map the stop frequencies """
	# df = feed.stop_frequencies()

	# # Create a Folium Map centered around the first point
	# first_point = df['geometry'].iloc[0]
	
	# mymap = folium.Map(location=[first_point.y, first_point.x], zoom_start=12)

	# # Add CircleMarkers for each point with size based on 'count'
	# for index, row in df.iterrows():
	#     point_coords = row['geometry']
	#     count_value = row['count']
	#     stop_id = row['stop_id']

	#     # Create a Popup with the 'count' value
	#     popup_content = f"Count: {count_value} \n Stop id: {stop_id} \n"
	#     popup = folium.Popup(popup_content, max_width=300)

	#     # Get the corresponding color from the color scale
	#     color_scale = folium.LinearColormap(colors=['blue', 'red'], vmin=df['count'].min(), vmax=df['count'].max())

	#     fill_color = color_scale(count_value)

	#     folium.CircleMarker(location=(point_coords.y, point_coords.x),
	#                         radius=count_value,  # Adjust the scale factor as needed
	#                         color='none',
	#                         fill=True,
	#                         fill_color=fill_color,
	#                         fill_opacity=0.6,
	#                         popup=popup).add_to(mymap)

	# color_scale.add_to(mymap)

	# # Add MeasureControl for the scale
	# mymap.add_child(MeasureControl(primary_length_unit='kilometers'))

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/stop_frequencies.html")

	# """ Visualize all trips """
	# print(feed.trips)

	# mymap = folium.Map(location=(feed.get_shape('1107D110').centroid.y, feed.get_shape('1107D110').centroid.x), zoom_start=12)

	# for index, row in feed.trips.iterrows():
	#     trip_id = row['trip_id']

	#     # Get the shape of the trip_id
	#     shape = feed.get_shape(trip_id) 

	#     # Extract coordinates from the LineString
	#     coordinates = list(shape.coords)

	#     # Create a Folium Map centered at the midpoint of the LineString
	#     midpoint = (shape.centroid.y, shape.centroid.x)		

	#     # Add the LineString to the map
	#     folium.PolyLine(locations=[[coord[1], coord[0]] for coord in shape.coords], color='blue').add_to(mymap)

	#     # Save the map to an HTML file
	#     mymap.save(f"{env.OUTPUT_PATH}/all_trips.html")	
	#     print(trip_id)

	"""
	3. Extract global metrics of the GTFS Feed
	"""
	# print(f"Simulation area: {feed.simulation_area_km2()} km2")
	# print(feed.trip_statistics())
	# print(feed.stop_statistics())

	# # Average distance between stops along the trips
	# dist = feed.ave_distance_between_stops_all(False)
	# weighted_average = np.average(dist['stop_dist_km'], weights=dist['n_stops'])

	# print(f"Average distance between stops: {weighted_average} km")

	"""
	4. Extract topological information 
	"""
	# print(feed.trip_crossovers())
	# print(feed.nearest_point_distance_km()) # Warning: takes a long time

	"""
	5. Operationnal metrics of a trip 
	"""
	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.2)

	# print(trip_sim.trip_duration_sec)
	# print(trip_sim.trip_length_km)

	# print(trip_sim.operation_estimates())
	# print(trip_sim.operation_estimates_aggregated())

	"""
	5. Operationnal metrics for the whole system
	"""
	# trips = feed.trips
	
	# df = pd.DataFrame()

	# for index, row in trips.iterrows():
	# 	trip_id = row['trip_id']
	# 	trip_sim = TripSim(feed = feed, trip_id=trip_id, ev_consumption = 0.4)

	# 	trip_stats = trip_sim.operation_estimates_aggregated()
	# 	df = pd.concat([df, pd.DataFrame([trip_stats])], ignore_index=True)
	
	# print(df)

	# print(df['n_trips'].sum())
	# print(df['vkm'].sum())
	# print(df['energy_kWh'].sum())

	# print(df['vkt'].mean())
	# print(df['energy_kWh_per_vehicle'].mean())

	"""
	6. Power/energy/speed profile of a single vehicle along a trip and associated statistics
	"""

	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.4)

	# time_values = np.arange(0, trip_sim.trip_duration_sec, 20) 
	# values = [trip_sim.power_profile(t) for t in time_values]
	# # values = [trip_sim.energy_profile(t) for t in time_values]
	# # values = [trip_sim.speed_profile(t) for t in time_values]

	# # Plot the function
	# plt.plot(time_values, values, marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	# print(trip_sim.vehicle_statistics())

	"""
	7. Profile of the vehicle fleet on a trip
	"""

	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.4)

	# trip_sim.simulate_vehicle_fleet(start_time = "05:00:00", stop_time = "22:00:00", time_step = 30, transient_state = False)

	# print(trip_sim.trip_profile)

	# # Plot the function
	# plt.plot(trip_sim.trip_profile['t'], trip_sim.trip_profile['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	"""
	8. Profile of a set of trips
	"""
	# trips = ['1107D110', '1107D111', '10114110']
	# ev_con = [0.4, 0.4, 0.4]

	# traffic_sim = TrafficSim(feed, trips, ev_con)

	# print(traffic_sim.operation_estimates().sum())
	# df = traffic_sim.profile(start_time = "05:00:00", stop_time = "23:00:00", time_step = 50, transient_state = True)

	# # Plot the function
	# plt.plot(df['t'], df['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	"""
	9. Profile of a the whole traffic network
	"""
	trips = list(feed.trips['trip_id'])
	print(trips)
	ev_con = [0.4] * len(trips)

	traffic_sim = TrafficSim(feed, trips, ev_con)

	print(traffic_sim.operation_estimates().sum())
	df = traffic_sim.profile(start_time = "05:00:00", stop_time = "23:00:00", time_step = 50, transient_state = True)

	# Plot the function
	plt.plot(df['t'], df['power_kW'], marker='o')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Power')
	plt.title('Power vs. Time')
	plt.grid(True)

	# plt.savefig('power_vs_time.png')
	plt.show()



	# trips = feed.trips['trip_id']

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
	# 	trip_sim = TripSim(feed = feed, trip_id=trip, ev_consumption = 0.4)

	# 	print(trip_sim.frequencies)

	# 	trip_sim.simulate_vehicle_fleet(time_span, time_step)


	# 	profile += trip_sim.trip_profile['power_kW'].values

	# print(profile)



	#print(feed.bounding_box())
	#print(feed.simulation_area_km2())

	#print(feed.stops)
	#feed.clean_all()
	#print(feed.stops)

	#df = feed.operation_estimates_all()
	#print(df)
	#print(df['vkm'].sum())
	#print(df['n_trips'].sum())

	# print(feed.nearest_point_distance_km()['trip_length_km'].mean)

	# print(feed.trip_statistics())

	# print(feed.trip_length_km_all()['trip_length_km'].mean)

	# print(feed.n_stops('20121111'))

	#dist = feed.ave_distance_between_stops_all(False)
	#weighted_average = np.average(dist['stop_dist_km'], weights=dist['n_stops'])

	#print(weighted_average)



	# print(feed.routes)
	# feed.clean_all()
	# print(feed.routes)
	# print(feed.trips)

	# print(feed.stops['stop_id'])

	#feed.clean_trips()


	"""
	Simulation
	"""

	# vehicle = Vehicle(feed = feed, trip_id='20121111', ev_consumption = 0.2)
	# print(vehicle.trip_data['distance'].sum())





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



	# time_values = np.arange(0, 4000, 20) 

	# vehicle = Vehicle(feed = feed, trip_id='20237110', ev_consumption = 0.4)

	# power_values = [vehicle.power_profile(t) for t in time_values]

	# # Plot the function
	# plt.plot(time_values, power_values, marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# plt.savefig('power_vs_time.png')
	# plt.show()




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