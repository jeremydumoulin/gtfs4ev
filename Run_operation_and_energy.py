# coding: utf-8

""" 
A python script that reproduces the operation and energy metrics for the 
different cities.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import os
import folium
from folium.plugins import MarkerCluster, MeasureControl, HeatMap
from folium.raster_layers import ImageOverlay
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter
from scipy.signal import convolve2d
from dotenv import load_dotenv
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
import csv

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim
from gtfs4ev.topology import Topology
from gtfs4ev import helpers as hlp

import Run_preprocess_gtfs as pp

"""
Parameters - PLEASE MODIFY ACCORDING TO YOUR NEEDS
"""

city = "Nairobi" # Used to name intermediate outputs and do some city-dependent pre-processing

# Input data
gtfs_feed_name = "GTFS_Nairobi" # Make sure the GTFS folder is in the input folder

# Parameters related GTFS Feed and traffic simulation
snap_to_osm_roads = False # Could take a long time. Data is generally already consistent with OSM network
ev_consumption = 0.4 # EV consumption (kWh/km) - Value should not affect the output
reuse_traffic_output = True # If True, serializes the dataframe with operationnal data in order to avopid recomputing TrafficSimulation

# Parameters related to timeseries
time_step = 50 # Time step in seconds

# Parameters related to baseline energy demand estimation
population = 5000000 # Total number of people
demand_per_capita = 400 # Yearly demand per capita (kWh)
active_working_days = 260 # Number of operating days a year of the minibus taxis

"""
Environment variables
"""
load_dotenv() # take environment variables from .env

INPUT_PATH = str(os.getenv("INPUT_PATH"))
OUTPUT_PATH = str(os.getenv("OUTPUT_PATH")) 

"""
Main code
"""

##########################################
# GTFS Feed Initialization & Preprocessing
##########################################

# Populate the feed with the raw data 
feed = GTFSFeed(gtfs_feed_name)

# Filter the feed according to city-specific rules defined in the preprocessing script
feed = pp.gtfs_preprocessing(feed, city)

# Clean and check data consistency
feed.clean_all() # Data cleaning to get a consistent feed
feed.check_all() # Re-check data consistency

# If necessary, snap the shapefiles to OSM road network (good to check once, but generally does)
if snap_to_osm_roads:
	feed.snap_shapes_to_osm() # Takes a lot of time

# Display general information about the data
feed.general_feed_info()
print(feed.simulation_area_km2())

##########################################################
# Traffic simulation & Operation estimates & Power Profile 
##########################################################

trips = list(feed.trips['trip_id']) # Trips to consider
ev_con = [ev_consumption] * len(trips) # List of EV consumption for all trips

# If True, only compute Traffic simulation if it has not already been done before
if reuse_traffic_output: 
	filename = f"{city}_tmp_operation_{ev_consumption}.pkl"

	# Check if a pickle file with same parameters already exists and the timeseries
	if os.path.exists(f"{OUTPUT_PATH}/{filename}") and os.path.exists(f"{OUTPUT_PATH}/{city}_powerprofile.csv"):
		print(f"INFO \t Using existing pickle data for operationnal simulation - Make sure it matches with the inputs")
		# Load the df from the file
		op = pd.read_pickle(f"{OUTPUT_PATH}/{filename}")  
	else:
		# Carry out the simulation
		traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
		op = traffic_sim.operation_estimates() # Get operation estimates
		# Serialize and save the df to a file
		op.to_pickle(f"{OUTPUT_PATH}/{filename}")
		df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = time_step, transient_state = False)
		df.to_csv(f"{OUTPUT_PATH}/{city}_powerprofile.csv", index = False)  
else:
	# Carry out the simulation
	traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
	op = traffic_sim.operation_estimates() # Get operation estimates

	# Timeseries
	df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = time_step, transient_state = False)
	df.to_csv(f"{OUTPUT_PATH}/{city}_powerprofile.csv", index = False)

####################
# Aggregated metrics
####################

print(f"Total energy demand per day (kWh): {op['energy_kWh'].sum()}")

print(f"Total VKM (km): {op['vkm'].sum()}")
print(f"Total number of vehicles: {op['ave_nbr_vehicles'].sum()}")
print(f"Average distance travelled by vehicle: {sum(op['vkt'] * op['ave_nbr_vehicles'])/op['ave_nbr_vehicles'].sum()}")

################################
# VKT per trip (distribution)
################################

# Different from the average distance by vehicle because not weigthed by the number 
# of vehicles per trip. Therefore, generally a bit higher because often they are only
# a small amount of trips yith very high VKT and/or they have a small number of vehicles
op['vkt'].to_csv(f"{OUTPUT_PATH}/{city}_vkt_per_trip.csv", index = False) 
print(f"Average distance travelled by vehicles on each trip: {op['vkt'].mean()}")

###############################
# Comparaison with local demand
###############################

print(f"Total yearly demand in kWh (pop x demand_per_capita): {population * demand_per_capita}")
print(f"Relative additionnal demand (%): {op['energy_kWh'].sum()*active_working_days/(population * demand_per_capita)*100}")


