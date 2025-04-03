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
import os
import time
import json
import pandas as pd
from shapely.ops import substring

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
    gtfs = GTFSManager(gtfs_datafolder="input/GTFS_Nairobi")

    # 1.2) Check the consistency of the GTFS data, and clean it if necessary
    # This step ensures that the data is valid for simulation
    if not gtfs.check_all():
        print("INFO: Data is inconsistent, cleaning data...")
        gtfs.clean_all()

    # 1.3) OPTIONAL - Additional filtering and data manipulation (uncomment to enable)
    # Example 1: Filter for services that run daily (e.g., remove weekend-only services)
    # gtfs.filter_services(service_id="DAILY") # Keep only trips operating under the specified service schedule.
    
    # Example 2: Filter out agencies that are not part of the bus fleet (e.g., exclude ferries)
    # gtfs.filter_agency(agency_id="UON") # Removes all data related to the specified agency.
    
    # Example 3: Add additional idle time at trip terminals (optional based on specific fleet simulation needs)
    # gtfs.add_idle_time(idle_time_seconds=60*30)  # Adds 30 minutes idle time at trip terminals

    # Example 4: Snap stops to trip shapes to improve consistency by ensuring stops are along the trip path
    # gtfs.snap_stops_to_tripshapes()

    # 1.4) OPTIONAL - Display general information and export GTFS summary
    # General information about the GTFS feed (e.g., number of trips, agencies, etc.)
    gtfs.general_feed_info()

    # Export summary statistics to a text file
    gtfs.export_statistics("output/GTFS_summary.txt")

    # OPTIONAL: Export a map of the GTFS data (e.g., stops, routes, and trips) as an HTML file
    # gtfs.to_map("output/GTFS_data.html")

    ###############################################################################
    ############# STEP 2: Simulate the operation of the vehicle fleet ############# 
    ###############################################################################

    # 2.1) Initialize the FleetSimulator with the GTFS data and a list of trip IDs to simulate
    # If no trip IDs are specified, all trips in the GTFS feed will be simulated
    fleet_sim = FleetSimulator(gtfs_manager=gtfs, trip_ids=["2017B111", "10106110"])
    # If you want to simulate all trips, uncomment the line below:
    # fleet_sim = FleetSimulator(gtfs_manager=gtfs)

    # 2.2) Compute the fleet operation for the selected trips
    # Use multiprocessing to speed up the simulation (set to False if you want a single-threaded computation)
    fleet_sim.compute_fleet_operation(use_multiprocessing=False)  # Set use_multiprocessing=True for parallel processing
    fleet_sim.fleet_operation.to_csv(f"output/fleet_operation.csv", index=False)

    # 2.3) OPTIONAL - Map and visualize the spatio-temporal movement of vehicles along a specific trip
    trip_id = "10106110"  # Replace with the trip ID you want to analyze

    # Initialize a TripSimulator to track the fleet movement along a specific trip
    tripsim = TripSimulator(gtfs_manager=gtfs, trip_id=trip_id)

    # Get the fleet trajectory (movement of vehicles) along the specified trip, with a time step of 120 seconds
    df = tripsim.get_fleet_trajectory(time_step=120)

    # Save the fleet trajectory to a CSV file for further analysis
    df.to_csv(f"output/{trip_id}_fleet_trajectory.csv", index=False)

    # 2.4) Visualize the fleet trajectory on a map
    vehicle_map = tripsim.map_fleet_trajectory(df)

    # Save the interactive map as an HTML file
    vehicle_map.save(f"output/{trip_id}_fleet_trajectory.html")

    # 2.5) Store the overall fleet operation results to a CSV file
    fleet_sim.fleet_operation.to_csv(f"output/{trip_id}_fleet_operation.csv", index=False)