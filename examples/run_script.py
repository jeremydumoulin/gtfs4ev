# coding: utf-8

"""
A Python script illustrating the basic usage of the GTFS4EV model for simulating and analyzing vehicle fleet operations.

This script demonstrates how to:
1. Load, clean, and filter GTFS data.
2. Simulate the operation of a vehicle fleet for a set of trips.
3. Optionally, visualize and store results of the fleet simulation.

Ensure the following structure for the input data:
- GTFS data folder (e.g., "input/GTFS_Nairobi")
- Output directories for results (e.g., "output/")
"""

import sys
import os, psutil
import time
import json
import pandas as pd
from shapely.ops import substring
import time
import gc

# Adding the parent directory to the Python path for access to the GTFS4EV module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Import relevant classes from the GTFS4EV package
from gtfs4ev.vehicle import Vehicle
from gtfs4ev.vehiclefleet import VehicleFleet
from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev.tripsimulator import TripSimulator
from gtfs4ev.fleetsimulator import FleetSimulator

if __name__ == "__main__":
    #################################################################################
    ########### STEP 1: Load, clean, and filter (optional) the GTFS data ###########
    #################################################################################

    # 1.1) Load GTFS data from the specified folder
    gtfs = GTFSManager(gtfs_datafolder="input/GTFS_Nairobi_cleaned")

    # 1.2) Check the consistency of the GTFS data, and clean it if necessary
    # This step ensures that the data is valid for simulation
    if not gtfs.check_all():
        print("INFO \t Data is inconsistent, cleaning data...")
        gtfs.clean_all()

    # 1.3) OPTIONAL - Additional filtering and data manipulation (uncomment to enable)
    # Example 1: Filter for services that run daily (e.g., remove weekend-only services)
    # gtfs.filter_services(service_id="DAILY") # Keep only trips operating under the specified service schedule.
    
    # Example 2: Filter out agencies that are not part of the bus fleet (e.g., exclude ferries)
    # gtfs.filter_agency(agency_id="UON") # Removes all data related to the specified agency.
    
    # Example 3: Add additional idle time at trip terminals (optional based on specific fleet simulation needs)
    gtfs.add_idle_time(idle_time_seconds=60*30)  # Adds 30 minutes idle time at trip terminals

    # Example 4: Trim tripshapes to make sure their start and end points correspond to the projection of the start
    # and stop stops locations once projected on the tripshape (needed later to calculate distance between stops)
    #gtfs.trim_tripshapes_to_terminal_locations()

    # 1.4) OPTIONAL - Display general information and export GTFS summary
    # Show general information about the GTFS feed (e.g., number of trips, agencies, etc.)
    gtfs.show_general_info()

    # Export cleaned/filtered GTFS data to GTFS file
    #gtfs.export_to_csv("input/GTFS_Nairobi_cleaned")

    # Export summary statistics to a text file
    gtfs.generate_summary_report("output/GTFS_summary.txt")

    # Export a map of a trip or the entire GTFS data (e.g., stops, routes, and trips) as an HTML file 
    #gtfs.generate_network_map("output/map_GTFS_data.html")
    trip_id = "1011F110"
    gtfs.generate_single_trip_map(trip_id = trip_id, filepath = f"output/GTFS_map_{trip_id}.html", projected = True)

    ###############################################################################
    ############# STEP 2: Simulate the operation of the vehicle fleet ############# 
    ###############################################################################

    # 2.1) Initialize the FleetSimulator with the GTFS data and a list of trip IDs to simulate
    # If no trip IDs are specified, all trips in the GTFS feed will be simulated
    fleet_sim = FleetSimulator(gtfs_manager=gtfs, trip_ids=["1011F110", "1107D110", "70002110"])
    # If you want to simulate all trips, uncomment the line below:
    #fleet_sim = FleetSimulator(gtfs_manager=gtfs)

    # 2.2) Compute the fleet operation for the selected trips
    # Use multiprocessing to speed up the simulation (set to False if you want a single-threaded computation)
    fleet_sim.compute_fleet_operation(use_multiprocessing=False)  # Set use_multiprocessing=True for parallel processing
    fleet_sim.fleet_operation.to_csv(f"output/Mobility_fleet_operation.csv", index=False)
    fleet_sim.trip_travel_sequences.to_csv(f"output/Mobility_trip_travel_sequences.csv", index=False)

    # # 2.3) OPTIONAL - Map and visualize the spatio-temporal movement of vehicles 
    # # Warning : this might take a very long time and a lot of disk space if many trips are simulated
    df = fleet_sim.get_fleet_trajectory(time_step=120)
    df.to_csv(f"output/Mobility_fleet_trajectory.csv", index=True)
    fleet_sim.generate_fleet_trajectory_map(fleet_trajectory=df, filepath=f"output/Mobility_fleet_trajectory_map.html")
    