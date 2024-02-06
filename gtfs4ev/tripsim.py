# coding: utf-8

""" 
TripSim
Simulates the behaviour of a single vehicle or a vehicle fleet along a trip and extracts relevant 
metrics. Is instantiated using a GTFSFeed, the trip_id, and the electric vehicle consumption (kWh/km). 
Provides both operational metrics and power/energy profiles. 
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform
import pyproj
from datetime import datetime, timedelta

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

from gtfs4ev.gtfsfeed import GTFSFeed

class TripSim:  

    """
    ATTRIBUTES
    """

    ev_consumption = 0.2 # Electric vehicle energy consumption (kWh/km)

    feed = "" # TrafficFeed object
    trip_id = "" # ID of the trip taken by the electric vehicle

    trip_data = pd.DataFrame() # Dataframe countaining all the relevant data for the vehicle trip

    trip_duration_sec = .0
    trip_length_km = .0

    frequencies = pd.DataFrame()

    # Simulation results
    trip_profile = pd.DataFrame()

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, feed, trip_id, ev_consumption):
        print(f"INFO \t Initializing a new vehicle with trip_id = {trip_id}", end="\r")

        self.set_ev_consumption(ev_consumption)

        self.set_feed(feed)
        self.set_trip_id(trip_id)
        
        self.set_trip_data()
        self.set_frequencies()

        self.set_trip_duration_sec()
        self.set_trip_length_km()        

    """ Setters """

    def set_ev_consumption(self, ev_consumption):
        """ Setter for ev_consumption attribute.
        Converts the value into a float
        """
        try:       
            ev_consumption = float(ev_consumption)
        except Exception as e:
            print(f"ERROR \t Impossible to convert the specified ev consumption into a float. - {e}")
        else:            
            self.ev_consumption = ev_consumption

    def set_feed(self, feed):
        """ Setter for feed attribute.
        Checks that the object is of the right type
        """
        try:       
            # Check if the value is an instance of the expected type
            if not isinstance(feed, GTFSFeed):
                raise TypeError(f"ERROR \t Expected an instance of GTFSFeed, but got {type(value)}")
        except TypeError:
            print(f"ERROR \t Impossible to initiate the traffic feed")
        else:            
            self.feed = feed

    def set_trip_id(self, trip_id):
        """ Setter for trip_id attribute
        """
        try:       
            # Check if the value exists
            row_trip_id = self.feed.trips[self.feed.trips['trip_id'] == trip_id].iloc[0]
        except Exception:
            print(f"ERROR \t Impossible to find the trip_id in the feed.")
        else:            
            self.trip_id = trip_id

    def set_trip_data(self):
        """ Setter for the trip data
        Creates a dataframe with all information to simulate the electric vehicle profiles
        """

        trip_id = self.trip_id

        # 1. Add the shape of the corresponding trip and project it into epsg:3857 crs 
        gdf = pd.merge(self.feed.trips, self.feed.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 

        linestring_web_mercator_projection = transform(web_mercator_projection, linestring)

        # 2. Create a table which contains the stop times for the trip, the stop name and geometry
        filtered_stop_times = self.feed.stop_times[self.feed.stop_times['trip_id'] == trip_id]        
        result_df = pd.merge(filtered_stop_times, self.feed.stops[['stop_id', 'stop_name', 'geometry']], on='stop_id', how='left') # Merge the filtered_df with the stop_points_df on 'stop_id'

        # 3. Add the closest point on the linestring path as the point for the stop
        result_df['closest_point_on_path'] = result_df['geometry'].apply(lambda point: hlp.find_closest_point(linestring, point))

        # 4. Create a dataframe that adds the driven segments to the stop sequence and assigns them a time, distance, speed, consumption, and average electrical power
        output_data = []

        time = 0
        cumulative_consumption = .0

        for i in range(0, len(result_df)):
            current_stop = result_df.iloc[i]               
            trip_id = current_stop['trip_id']

            if i == 0:
                # Only Add the information for the current stop 

                duration = (current_stop['departure_time']-current_stop['arrival_time']).total_seconds()
                time += duration 

                output_data.append({
                    'trip_id': trip_id,
                    'moving': False,
                    'departure_time': current_stop['departure_time'],
                    'arrival_time': current_stop['arrival_time'],
                    'start_stop_name': current_stop['stop_name'],
                    'end_stop_name': current_stop['stop_name'],
                    'start_stop_loc': current_stop['closest_point_on_path'],
                    'end_stop_loc': current_stop['closest_point_on_path'],                    
                    'time': time,
                    'duration': duration,
                    'distance': .0,
                    'speed': .0,
                    'power': .0,
                    'consumption': .0,
                    'cumulative_consumption': .0
                })

            # Add the information for the segment travelled between stops and the current stop
            else:             
                previous_stop = result_df.iloc[i - 1]

                # Define the projection from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)                
                start_point = transform(web_mercator_projection, previous_stop['closest_point_on_path'])
                end_point = transform(web_mercator_projection, current_stop['closest_point_on_path'])

                # Calculate the segment indicators 
                duration = (current_stop['arrival_time']-previous_stop['departure_time']).total_seconds()
                distance_along_line = abs(linestring_web_mercator_projection.project(start_point) - linestring_web_mercator_projection.project(end_point)) 
                speed = distance_along_line / duration * 3.6 # Average speed in km/h
                consumption = distance_along_line * 1e-3 * self.ev_consumption # consumption in kWh
                power = consumption * 3.6e3 / duration # Electrical power in kW

                time += duration
                cumulative_consumption += consumption

                output_data.append({
                    'trip_id': trip_id,
                    'moving': True,
                    'departure_time': previous_stop['departure_time'],
                    'arrival_time': current_stop['arrival_time'],
                    'start_stop_name': previous_stop['stop_name'],
                    'end_stop_name': current_stop['stop_name'],
                    'start_stop_loc': previous_stop['closest_point_on_path'],
                    'end_stop_loc': current_stop['closest_point_on_path'],
                    'time': time,
                    'duration': duration,
                    'distance': distance_along_line,
                    'speed': speed,
                    'power': power,
                    'consumption': consumption,
                    'cumulative_consumption': cumulative_consumption
                })

                duration = (current_stop['departure_time']-current_stop['arrival_time']).total_seconds()

                time += duration

                # Add the information for the current stop 
                output_data.append({
                    'trip_id': trip_id,
                    'moving': False,
                    'departure_time': current_stop['departure_time'],
                    'arrival_time': current_stop['arrival_time'],
                    'start_stop_name': current_stop['stop_name'],
                    'end_stop_name': current_stop['stop_name'],
                    'start_stop_loc': current_stop['closest_point_on_path'],
                    'end_stop_loc': current_stop['closest_point_on_path'],
                    'time': time,
                    'duration': duration,
                    'distance': .0,
                    'speed': .0,
                    'power': .0,
                    'consumption': .0,
                    'cumulative_consumption': cumulative_consumption
                })        


            output_df = pd.DataFrame(output_data)

            self.trip_data = output_df

    def set_trip_duration_sec(self):
        """ Setter for duration
        """                
        self.trip_duration_sec = self.feed.trip_duration_sec(self.trip_id)

    def set_trip_length_km(self):
        """ Setter for length
        """                
        self.trip_length_km = self.feed.trip_length_km(self.trip_id)

    def set_frequencies(self):
        """ Setter for trip_frequencies attribute
        """
        frequencies = self.feed.frequencies[self.feed.frequencies['trip_id'] == self.trip_id].copy()
        frequencies = frequencies.reset_index(drop=True)
        self.frequencies = frequencies

    """ Statistics and profiles of a single vehicle """

    def power_profile(self, t, loop = False):  

        if t > self.trip_data['time'].iloc[-1]:
            if not loop:
                return 0
            else:
                t = t % self.trip_data.iloc[-1]['time']
                
        nearest_higher_index = (self.trip_data['time'] - t).apply(lambda x: float('inf') if x <= 0 else x).idxmin()            

        return self.trip_data.loc[nearest_higher_index, 'power']

    def energy_profile(self, t, loop = False):       

        if t > self.trip_data['time'].iloc[-1]:
            if not loop:
                return 0
            else:
                t = t % self.trip_data.iloc[-1]['time']

        nearest_higher_index = (self.trip_data['time'] - t).apply(lambda x: float('inf') if x <= 0 else x).idxmin()

        return self.trip_data.loc[nearest_higher_index, 'cumulative_consumption']

    def speed_profile(self, t, loop = False):
        nearest_higher_index = (self.trip_data['time'] - t).apply(lambda x: float('inf') if x <= 0 else x).idxmin()

        if t > self.trip_data['time'].iloc[-1]:
            if not loop:
                return 0
            else:
                t = t % self.trip_data.iloc[-1]['time']

        nearest_higher_index = (self.trip_data['time'] - t).apply(lambda x: float('inf') if x <= 0 else x).idxmin()

        return self.trip_data.loc[nearest_higher_index, 'speed']

    def vehicle_statistics(self):     
        statistics = {
            'trip_duration_s': self.trip_data.iloc[-1]['time'],
            'total_consumption_kWh': self.trip_data.iloc[-1]['cumulative_consumption'],
            'total_distance_m': self.trip_data['distance'].sum(),
            'total_stop_time': self.trip_data[~self.trip_data['moving']]['duration'].sum(),
            'total_moving_time': self.trip_data[self.trip_data['moving']]['duration'].sum(),
            'average_speed': (self.trip_data['speed'] * self.trip_data['duration']).sum() / self.trip_data['duration'].sum(),
            'average_speed_excluding_stop_times': (self.trip_data[self.trip_data['moving']]['speed'] * self.trip_data[self.trip_data['moving']]['duration']).sum() / self.trip_data[self.trip_data['moving']]['duration'].sum(),
            'average_power': (self.trip_data['power'] * self.trip_data['duration']).sum() / self.trip_data['duration'].sum(),
            'average_power_excluding_stop_times': (self.trip_data[self.trip_data['moving']]['power'] * self.trip_data[self.trip_data['moving']]['duration']).sum() / self.trip_data[self.trip_data['moving']]['duration'].sum(),
            'average_distance_btw_stops': self.trip_data[self.trip_data['moving']]['distance'].mean()
        }
        
        return statistics


    """ Simulation of operation estimates of the trip """

    def operation_estimates(self):
        frequencies = self.frequencies
        trip_id = self.trip_id
        trip_length_km = self.trip_length_km
        trip_duration_sec = self.trip_duration_sec

        df = pd.DataFrame(frequencies)

        df['trip_id'] = trip_id

        df['timeslot_duration_sec'] = (df['end_time'] - df['start_time']).dt.total_seconds()

        df['n_vehicles'] = trip_duration_sec/df['headway_secs']
        #df['n_vehicles'] = np.where(df['n_vehicles'] == 0, 1, df['n_vehicles']) # Put 1 if the output is 0

        df['trips_per_vehicle'] = df['timeslot_duration_sec'] / trip_duration_sec

        df['n_trips'] = df['trips_per_vehicle'] * df['n_vehicles'] 

        df['vkm'] = df['trips_per_vehicle'] * df['n_vehicles'] * trip_length_km

        df['vkt'] = df['vkm'] / df['n_vehicles']

        df['energy_kWh'] = df['vkm'] * self.ev_consumption

        df['energy_kWh_per_vehicle'] = df['vkt'] * self.ev_consumption

        return df

    def operation_estimates_aggregated(self):
        df = self.operation_estimates()

        n_trips = df['n_trips'].sum()
        vkm = df['vkm'].sum()
        ave_nbr_vehicles = np.average(df['n_vehicles'], weights=df['timeslot_duration_sec'])

        trips_per_vehicle = df['trips_per_vehicle'].sum()
        vkt = df['vkt'].sum()

        energy = df['energy_kWh'].sum()
        energy_per_vehicle = df['energy_kWh_per_vehicle'].sum()

        statistics = {
            'trip_id': self.trip_id,
            'n_trips': n_trips,
            'vkm': vkm,
            'ave_nbr_vehicles': ave_nbr_vehicles,
            'trips_per_vehicle': trips_per_vehicle,
            'vkt': vkt,
            'energy_kWh':energy,
            'energy_kWh_per_vehicle':energy_per_vehicle
        }

        return statistics


    def simulate_vehicle_fleet(self, start_time, stop_time, time_step, transient_state = False):

        # 1. Check the start and stop times and extract the duration of the simulation

        # Parse input strings into datetime objects
        start_datetime = datetime.strptime(start_time, "%H:%M:%S")
        stop_datetime = datetime.strptime(stop_time, "%H:%M:%S")

        # Check conditions
        if start_datetime >= datetime.strptime("00:00:00", "%H:%M:%S") and stop_datetime <= datetime.strptime("23:59:59", "%H:%M:%S") and start_datetime < stop_datetime:
            # Calculate duration
            duration = (stop_datetime - start_datetime).total_seconds()
        else:
            print("ERROR \t Stop time must be greater than the start_time")
            return None

        # 2. Initialize the output data variables      

        output_data = []

        energy = .0
        power = .0
        n_vehicles = .0
        n_vehicles_previous = 1

        # 3. Get the operation estimates and other usefull values

        df = self.operation_estimates()

        min_datetime = df['start_time'].iloc[0]
        max_datetime = df['end_time'].iloc[-1]

        # 3. Calculate the power and cumulated energy for the vehicle fleet for each time step
        # The idea is to find the corresponding headway_sec in order to calculate the outputs. If there is no time slot with a headway sec for the datetime, the output is zero
        # If the transient state has to be calculated, then take into account the rise of the vehicle fleet of the current time slot and the decay of the previous time slot
        # If we are above the datetime of the last time slot, also assess the decay of vehicles of the later

        time_values = np.arange(0, duration, time_step)

        for t in time_values:
            # Get the current datetime and reset the power            
            current_datetime = start_datetime + timedelta(seconds=t)
            power = .0
            decay_time = 0

            # If the transient state is to be calculated, calculate the decay time of the last time slot
            if transient_state:
                decay_time = df['n_vehicles'].iloc[-1]*df['headway_secs'].iloc[-1]

            # If the current datime has no running vehicles, set the power and the additionnal energy to 0
            # Else assess the power and energy of the vehicle fleet
            mask = (df['start_time'] <= current_datetime) & (current_datetime <= df['end_time']) 

            if (current_datetime < min_datetime) or (current_datetime > (max_datetime + timedelta(seconds=decay_time)) ):
                power = .0
                energy += .0
            else:                
                # Get the index of the row with the relevant operational information            
                # If in the decay time of the last time slot, set to the index of the last time slot
                if transient_state and current_datetime > max_datetime:
                    index = df.tail(1).index[0]
                else:
                    index = df[mask].index[0]
                
                # Get the corresponding headway_sec and the elapsed time between the beginning of the time slot and the current time (local time)    
                headway_sec = df.loc[index, 'headway_secs']
                t_local = (current_datetime - df.loc[index, 'start_time']).total_seconds()

                # Assess the number of vehicles - must be an integer                
                # If the transient state has to be calculated, the number of vehicles of the current time slot is increasing over time 
                # If the result is 0, set the number of vehicles to 1
                if transient_state and ((t_local - self.trip_duration_sec) < 0):
                    n_vehicles = max(round(t_local / headway_sec), 1) 
                else:
                    n_vehicles = max(round(df.loc[index, 'n_vehicles']), 1)
     
                # Loop over the vehicles
                # For each vehicle, add the power for the local time delayed by the headway time 
                if current_datetime <= max_datetime:
                    for vehicle_index in range(n_vehicles):                                  
                        power += self.power_profile(t_local + vehicle_index * headway_sec, loop=True)                  

                # Add also the decay of the fleet of the previous time slot if the transient state neeeds to be calculated
                if transient_state and (index > 0 or current_datetime > max_datetime):
                    # If we are in the datetime of the vehicles of the last time slot, give a dummy index to ensure the code works  
                    if current_datetime > max_datetime:
                        index = index + 1
                        t_local = (current_datetime - df.loc[index-1, 'end_time']).total_seconds()                                           

                    headway_sec = df.loc[index-1, 'headway_secs']                    
                    n_vehicles_previous = int(round(df.loc[index-1, 'n_vehicles']))            

                    if (t_local - self.trip_duration_sec) < 0 :
                        n_vehicles_previous = n_vehicles_previous - round(t_local / headway_sec)
                        
                    else:
                        n_vehicles_previous = 0

                    for vehicle_index in range(n_vehicles_previous):                                  
                        power += self.power_profile(t_local + vehicle_index * headway_sec, loop=True)

                energy += power * time_step / 3600                    
  
            output_data.append({
                't': t,
                'power_kW': power,
                'energy_kWh': energy,
                'n_vehicles': n_vehicles
            })  

        output_df = pd.DataFrame(output_data) 

        self.trip_profile = output_df


    def energy_profile_trip(self, t):

        integrand = self.power_profile_trip

        # Integrate the power profile from 0 to t
        result, _ = quad(integrand, 0, t)

        return result
