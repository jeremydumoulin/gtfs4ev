# coding: utf-8

""" 
A python script that reproduces that calculates the economic savings for the 
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
reuse_traffic_output = True # If True, serializes the dataframe with operationnal data in order to avopid recomputing TrafficSimulation
active_working_days = 260 # Number of operating days a year of the minibus taxis

# Parameters related to Diesel cost
diesel_consumption = 0.1 # Diesel consumption (L/km)
diesel_price = 1.5 # Diesel price (US$/L)
diesel_subsidies = 0.1 # Diesel explicit subsidies (US$/L)

# Parameters related to EV cost
ev_consumption = 0.4 # EV consumption (kWh/km) - Value should not affect the output
charging_efficiency = 0.9 
electricity_price = 0.15 # Electricity price (US$/kWh)

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

##########################################
# Traffic simulation & Operation estimates 
##########################################

trips = list(feed.trips['trip_id']) # Trips to consider
ev_con = [ev_consumption] * len(trips) # List of EV consumption for all trips

# If True, only compute Traffic simulation if it has not already been done before
if reuse_traffic_output: 
	filename = f"{city}_tmp_operation_{ev_consumption}.pkl"

	# Check if a pickle file with same parameters already exists and the timeseries
	if os.path.exists(f"{OUTPUT_PATH}/{filename}"):
		print(f"INFO \t Using existing pickle data for operationnal simulation - Make sure it matches with the inputs")
		# Load the df from the file
		op = pd.read_pickle(f"{OUTPUT_PATH}/{filename}")  
	else:
		# Carry out the simulation
		traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
		op = traffic_sim.operation_estimates() # Get operation estimates
		# Serialize and save the df to a file
		op.to_pickle(f"{OUTPUT_PATH}/{filename}")
else:
	# Carry out the simulation
	traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
	op = traffic_sim.operation_estimates() # Get operation estimates

#############################
# Per vehicle metrics savings
#############################

n_vehicles = op['ave_nbr_vehicles'].sum()
vkm = op['vkm'].sum()
distance_per_vehicle = sum(op['vkt'] * op['ave_nbr_vehicles'])/op['ave_nbr_vehicles'].sum()

savings_per_km = (diesel_consumption * diesel_price) - (ev_consumption / charging_efficiency * electricity_price)
savings_per_km_without_subsidies = (diesel_consumption * (diesel_price+diesel_subsidies)) - (ev_consumption / charging_efficiency * electricity_price)

print(f"Average daily driven distance per vehicle (km): {distance_per_vehicle}")
print(f"Savings per km w/ subsidies ($/km): {savings_per_km}")
print(f"Savings per km w/o subsidies ($/km): {savings_per_km_without_subsidies}")

print(f"Average per vehicle savings w/ subsidies (US$/year): {distance_per_vehicle*active_working_days*savings_per_km}")
print(f"Average per vehicle savings w/o subsidies (US$/year): {distance_per_vehicle*active_working_days*savings_per_km_without_subsidies}")


