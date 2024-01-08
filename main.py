# coding: utf-8

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj

from gtfs4ev.vehicle import Vehicle
from gtfs4ev.trafficfeed import TrafficFeed
from gtfs4ev import helpers as hlp

"""
Main function
"""
def main():

	feed = TrafficFeed("GTFS_Nairobi")

	#feed.general_feed_info()

	# print(f"min_lat: {feed.min_latitude}")	
	# print(f"max_lat: {feed.max_latitude}")	
	# print(f"min_lon: {feed.min_longitude}")	
	# print(f"max_lon: {feed.max_longitude}")	

	# print(f"Bounding box: {feed.bounding_box}")	

	#print(feed.simulation_area())

	trip_id = "1107D110"

	# Get the shape corresponding to the trip

	gdf = pd.merge(feed.trips, feed.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
	linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

	# Get the stops of the trip with the corresponding latitude and longitude
	
	filtered_stop_times = feed.stop_times[feed.stop_times['trip_id'] == trip_id]

	# Merge the filtered_df with the stop_points_df on 'stop_id'
	result_df = pd.merge(filtered_stop_times, feed.stops[['stop_id', 'stop_name', 'geometry']], on='stop_id', how='left')

	# Add the closest point on the path as the point for the stop
	result_df['closest_point'] = result_df['geometry'].apply(lambda point: hlp.find_closest_point(linestring, point))
	
	# Initialize an empty list to store the data
	new_data = []

	# Iterate over the DataFrame to get the departure and arrival times between two consecutive stops
	# Skip the first row
	for i in range(1, len(result_df)):
	    current_stop = result_df.iloc[i]
	    previous_stop = result_df.iloc[i - 1]

	    # Extract relevant information
	    trip_id = current_stop['trip_id']
	    departure_time = previous_stop['departure_time']
	    arrival_time = current_stop['arrival_time']
	    start_stop_name = previous_stop['stop_name']
	    end_stop_name = current_stop['stop_name']
	    start_stop_loc = previous_stop['closest_point']
	    end_stop_loc = current_stop['closest_point']

	    # Define the projection from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
	    web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 
	    line_shape_web_mercator = transform(web_mercator_projection, linestring)
	    point1_web_mercator = transform(web_mercator_projection, start_stop_loc)
	    point2_web_mercator = transform(web_mercator_projection, end_stop_loc)

	    distance_along_line = abs(line_shape_web_mercator.project(point1_web_mercator) - line_shape_web_mercator.project(point2_web_mercator))

	    ave_speed = distance_along_line / (arrival_time-departure_time).total_seconds() * 3.6 # Average speed in km/h

	    # Append the data to the list
	    new_data.append({
	        'trip_id': trip_id,
	        'departure_time': departure_time,
	        'arrival_time': arrival_time,
	        'start_stop_name': start_stop_name,
	        'end_stop_name': end_stop_name,
	        'start_stop_loc': start_stop_loc,
	        'end_stop_loc': end_stop_loc,
	        'distance': distance_along_line,
	        'average_speed': ave_speed
	    })

	# Create a new DataFrame from the list
	new_df = pd.DataFrame(new_data)

	# Print the new DataFrame
	print(new_df)
		

if __name__ == "__main__":
	main()