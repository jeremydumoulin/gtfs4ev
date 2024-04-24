# coding: utf-8

""" 
A python script that reproduces TRAP (Traffic Related Air Pollution) exposure results 
for a given city by cross-referencing GTFS and GIS population data (a population layer 
covering the area is required).
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
population_raster_name = "Nairobi_GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R10_C22.tif" # Make sure it is in the input folder as defined in the .env file
gtfs_feed_name = "GTFS_Nairobi" # Make sure the GTFS folder is in the input folder

# Parameters related GTFS Feed and traffic simulation
snap_to_osm_roads = False # Could take a long time. Data is generally already consistent with OSM network
ev_consumption = 0.4 # EV consumption (kWh/km)
reuse_traffic_output = True # If True, serializes the dataframe with operationnal data in order to avopid recomputing TrafficSimulation

# Parameters related to TRAP exposure assessments
decay_rate = 0.0
buffer_distance = 300 # Distance in meters

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

#############################################################
# Traffic simulation & Operation estimates of the whole fleet
#############################################################

trips = list(feed.trips['trip_id']) # Trips to consider
ev_con = [ev_consumption] * len(trips) # List of EV consumption for all trips

# If True, only compute Traffic simulation if it has not already been done before
if reuse_traffic_output: 
	filename = f"{city}_tmp_air_pollution_exposure_{ev_consumption}_operation.pkl"

	# Check if a pickle file with same parameters already exists
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

# Get the main metrics needed for TRAP exposure calculation
vkm_list = op['vkm'].tolist() # VKM of trips
linestring_list = [feed.get_shape(row['trip_id']) for index, row in op.iterrows()] # Associated linestrings