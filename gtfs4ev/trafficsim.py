# coding: utf-8

""" 
TrafficSim
Simulates the behaviour of a vehicle fleet along a set of several trips. Is instanciated using a 
GTFSFeed, a list of trip_ids, and a list of corresponding electric vehicle consimption (kWh/km). 
Provides operational metrics and profiles for the set of trips.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform
import pyproj

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim

class TrafficSim:  

    """
    ATTRIBUTES
    """

    feed = "" # TrafficFeed object
    trip_simulations = []

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, feed, trip_ids, ev_consumptions):
        self.set_feed(feed)
        self.set_trip_simulations(trip_ids, ev_consumptions)        

    """ Setters """

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

    def set_trip_simulations(self, trip_ids, ev_consumptions):
        """ 
        """
        try:     
            if not len(trip_ids) == len(ev_consumptions):
                raise Exception()
        except Exception:
            print(f"Error: The length of the ev_consumptions must be equal to the length of trip_ids")
        else:            
            for i in range(0, len(trip_ids)):
                trip_sim = TripSim(feed = self.feed, trip_id = trip_ids[i], ev_consumption = ev_consumptions[i])
                self.trip_simulations.append(trip_sim)

    """ Operation estimates of the trips """

    def operation_estimates(self):
        df = pd.DataFrame()

        # Iterate over rows of df1 and append rows to df2
        for trip in self.trip_simulations:
            operation_estimates = trip.operation_estimates_aggregated()
            df = pd.concat([df, pd.DataFrame([operation_estimates])], ignore_index=True)

        return df

    """ Power and energy profile of the trips """

    def profile(self, start_time, stop_time, time_step, transient_state = False):
        df = pd.DataFrame()

        trip = self.trip_simulations

        print("Calculating the power and energy profile for the whole trips:")

        trip[0].simulate_vehicle_fleet(start_time, stop_time, time_step, transient_state)
        df['t'] = trip[0].trip_profile['t']
        df['power_kW'] = trip[0].trip_profile['power_kW'] 
        df['energy_kWh'] = trip[0].trip_profile['energy_kWh']   

        # Iterate over rows of df1 and append rows to df2
        i = 1
        for trip in self.trip_simulations[1:]:
            print(f"\t Completion: {i / len(self.trip_simulations) * 100}%")
            trip.simulate_vehicle_fleet(start_time, stop_time, time_step, transient_state)

            df['power_kW'] += trip.trip_profile['power_kW']
            df['energy_kWh'] += trip.trip_profile['energy_kWh']
            i = i+1           

        return df
    
