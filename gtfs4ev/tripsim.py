# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform
import pyproj
from datetime import datetime

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
        print(f"Initializing a new vehicle with trip_id = {trip_id}")

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
            print(f"\t Error: Impossible to convert the specified ev consumption into a float. - {e}")
        else:            
            self.ev_consumption = ev_consumption

    def set_feed(self, feed):
        """ Setter for ev_consumption attribute.
        Checks that the object is of the right type
        """
        try:       
            # Check if the value is an instance of the expected type
            if not isinstance(feed, GTFSFeed):
                raise TypeError(f"Expected an instance of YourCustomType, but got {type(value)}")
        except TypeError:
            print(f"\t Error: Impossible to initiate the traffic feed")
        else:            
            self.feed = feed

    def set_trip_id(self, trip_id):
        """ Setter for trip_id attribute
        """
        try:       
            # Check if the value exists
            row_trip_id = self.feed.trips[self.feed.trips['trip_id'] == trip_id].iloc[0]
        except Exception:
            print(f"\t Error: Impossible to find the trip_id in the feed.")
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


    def simulate_vehicle_fleet(self, duration, time_step):

        time_values = np.arange(0, duration, time_step)

        output_data = []

        energy = .0
        power = .0
        speed = .0
        n_stopped = 0
        n_moving = 0
        tot_speed = .0

        for value in time_values:
            t = value
            power = .0
            tot_speed = .0
            n_moving = n_stopped = 0

            if t > self.frequencies['time'].iloc[-1]:
                print("Error: specified timeframe is greater than the total trip time")
                power = .0
                energy += .0
                speed = .0
                tot_speed = .0
                n_moving = n_stopped = 0
            else:     
                index = (self.frequencies['time'] - t).apply(lambda x: float('inf') if x <= 0 else x).idxmin()            

                if index != self.frequencies.index[0]:
                    t = t - self.frequencies.loc[index-1, 'time']                
                
                for vehicle_index in range(self.frequencies.loc[index, 'n_vehicles']):
                    # print(vehicle_index)                
                    power += self.power_profile(t + vehicle_index * self.frequencies.loc[index, 'headway_secs'], loop=True)
                    speed = self.speed_profile(t + vehicle_index * self.frequencies.loc[index, 'headway_secs'], loop=True)
                    tot_speed += speed
                    if speed == .0:
                        n_stopped += 1
                    
                speed = tot_speed / self.frequencies.loc[index, 'n_vehicles']    
                energy += power * time_step / 3600
                n_moving = self.frequencies.loc[index, 'n_vehicles'] - n_stopped

            output_data.append({
                't': value,
                'power_kW': power,
                'energy_kWh': energy,
                'average_speed_kmh': speed,
                'nbr_vehicles_moving': n_moving,
                'nbr_vehicles_stopped': n_stopped
            })  

        output_df = pd.DataFrame(output_data) 

        self.trip_profile = output_df


    def energy_profile_trip(self, t):

        integrand = self.power_profile_trip

        # Integrate the power profile from 0 to t
        result, _ = quad(integrand, 0, t)

        return result
