# coding: utf-8

""" 
A python script illustrating the basic usage of the GTFS4EV model.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory

from gtfs4ev.vehicle import Vehicle
from gtfs4ev.vehiclefleet import VehicleFleet
from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev.tripsimulator import TripSimulator

# STEP 1: Define the electric vehicle fleet
# Create instances of Vehicle representing different types with specific attributes (e.g., battery capacity, consumption rate)

minibus = Vehicle(name="minibus", battery_capacity_kWh=70, consumption_kWh_per_km=0.39)
bus = Vehicle(name="bus", battery_capacity_kWh=200, consumption_kWh_per_km=0.5, max_charging_power_kW=1) # Define max charging power for motorcycles

# Create a fleet of vehicles, specifying proportions for each vehicle type (e.g., 90% cars, 10% motorcycles)
fleet = VehicleFleet(vehicle_types=[[minibus, 0.9], [bus, 0.1]])

print(f"\t Average battery capacity: {fleet.average_battery_capacity()} kWh" )
print(f"\t Average electric consumption: {fleet.average_consumption()} kWh/km")

# STEP 2: Load, clean, and filter (optionnal) the GTFS data
# Create an instance of the GTFSManager data, check consistency and show some general transit indicators

gtfs = GTFSManager(gtfs_datafolder = "C:/Users/dumoulin/Documents/_CODES/gtfs4ev/examples/input/GTFS_Nairobi")

# Check data consistency and perform a cleaning if needed
if not gtfs.check_all():
	gtfs.clean_all()

# Filtering (Optional) - Allows data refinement (e.g.: exclude weekends, remove agencies that are not part of the bus fleet such as ferries).
# gtfs.filter_agency(agency_id="UON")   # Removes all data related to the specified agency.
# gtfs.filter_services(service_id="DAILY")  # Keeps only trips operating under the specified service schedule.

gtfs.general_feed_info()

# gtfs.to_map("output/GTFS_data.html")
# gtfs.export_statistics("output/GTFS_summary.txt")

print(gtfs.trip_length_km(trip_id = "10106110"))

# STEP 2: Simulate the operation of the electric buses

tripsim = TripSimulator(gtfs_manager = gtfs, trip_id = "10106110")

print(tripsim.trip_duration_sec())
print(tripsim.max_vehicles_in_operation())

df = tripsim.simulate_fleet_operation(time_step=60*2)


print("creating a map")

vehicle_map = tripsim.create_map_with_slider()
vehicle_map.save("vehicle_movement.html")


