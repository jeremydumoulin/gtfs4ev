# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

class TrafficFeed:  

    """
    ATTRIBUTES
    """
    
    datapath = "" # Absolute path to the GTFS datafolder

    # Panda dataframes holding standard data
    agency = pd.DataFrame()    
    routes = pd.DataFrame()
    stop_times = pd.DataFrame()
    calendar = pd.DataFrame()
    frequencies = pd.DataFrame()
    trips = pd.DataFrame()

    # Geopanda dataframes holding georeferenced data
    shapes = gpd.GeoDataFrame(columns=['shape_id', 'geometry'], crs="EPSG:4326")
    stops = gpd.GeoDataFrame(columns=['stop_id', 'stop_name', 'geometry'], crs="EPSG:4326")   

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, gtfs_foldername):
        print(f"Initializing the TrafficFeed object using the /{gtfs_foldername} data folder: ")

        self.set_datapath(gtfs_foldername)

        print("\t -")

        self.set_agency()
        self.set_routes()
        self.set_stop_times()
        self.set_calendar()
        self.set_frequencies()
        self.set_trips()

        self.set_shapes()
        self.set_stops()

    """ Setters """

    def set_datapath(self, gtfs_foldername):
        """ Setter for datafolder attribute.
        Checks if the GTFS datafolder exists and contains all the required files
        """
        try:
            abs_path = env.INPUT_PATH / str(gtfs_foldername)
            files_to_check = ['agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt', 'trips.txt']          

            if not os.path.isdir(abs_path):
                raise FileNotFoundError() 

            try:                
                for file_name in files_to_check:
                    file_path = os.path.join(abs_path, file_name)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError()                    
            except Exception:
                print("\t Error: one of the required GTFS files seems to be missing. Make sure the following files are in the data folder: 'agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt'.")
            finally:
                print("\t Folder found. All required .txt files are present.")
                for file_name in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file_name)
                    if file_name not in files_to_check:
                        print(f"\t Warning: The data folder contains an additional but unused file named '{file_name}'.")

        except FileNotFoundError as e:
            print(f"\t Error: unable to open the /{gtfs_foldername} folder. Make sure the data folder exists. ")
        except Exception as e:
            print(f"\t Error: {e}")
        else:            
            self.datapath = abs_path

    def set_agency(self):
        """ Setter for agency attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Agency: ", end="", flush=True)
        
        file_path = open(self.datapath / "agency.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['agency_id', 'agency_name', 'agency_url', 'agency_timezone']
        column_types = {'agency_id': str, 'agency_name': str, 'agency_url': str, 'agency_timezone': str}
            
        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)
        except Exception:
            print("Error: it seems that some of the required columns of the agency.txt file are missing. Please check 'agency_id', 'agency_name', 'agency_url', 'agency_timezone' are present. ")    
        else:
            self.agency = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_routes(self):
        """ Setter for routes attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Routes: ", end="", flush=True)
        
        file_path = open(self.datapath / "routes.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'agency_id', 'route_short_name', 'route_long_name']
        column_types = {'route_id': str, 'agency_id': str, 'route_short_name': str, 'route_long_name': str}            

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)       
        except Exception:
            print("Error: it seems that some of the required columns of the routes.txt file are missing. Please check 'route_id', 'agency_id', 'route_short_name', 'route_long_name' are present. ")    
        else:
            self.routes = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_stop_times(self):
        """ Setter for stop_times attribute.
        Keeps only the five required columns of the GTFS standard
        """
        print("\t Stop times: ", end="", flush=True)

        file_path = open(self.datapath / "stop_times.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
        column_types = {'trip_id': str, 'arrival_time': str, 'departure_time': str, 'stop_id': str, 'stop_sequence': int}     

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)
            df['arrival_time']= pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
            df['departure_time']= pd.to_datetime(df['departure_time'], format='%H:%M:%S')                
        except Exception:
            print("Error: it seems that some of the required columns of the stop_times.txt file are missing. Please check 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence' are present. ")    
        else:
            self.stop_times = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_calendar(self):
        """ Setter for calendar attribute.
        """
        print("\t Calendar: ", end="", flush=True)
        
        file_path = open(self.datapath / "calendar.txt", "r", encoding = "utf-8")       
        
        column_types = {'service_id': str, 'monday': bool, 'tuesday': bool, 'wednesday': bool, 'thursday': bool, 'friday': bool, 'saturday': bool, 'sunday': bool, 'start_date': str, 'end_date': str}     

        try:          
            df = pd.read_csv(file_path, dtype=column_types)
            df['start_date']= pd.to_datetime(df['start_date'], format='%Y%m%d')
            df['end_date']= pd.to_datetime(df['end_date'], format='%Y%m%d')                
        except Exception:
            print("Error: it seems that some of the required columns of the calendar.txt file are missing. ")    
        else:
            self.calendar = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_frequencies(self):
        """ Setter for frequencies attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Frequencies: ", end="", flush=True)

        file_path = open(self.datapath / "frequencies.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['trip_id', 'start_time', 'end_time', 'headway_secs']
        column_types = {'trip_id': str, 'start_time': str, 'end_time': str, 'headway_secs': int}            

        try:          
            df = pd.read_csv(file_path, dtype=column_types)
            df['start_time']= pd.to_datetime(df['start_time'], format='%H:%M:%S')
            df['end_time']= pd.to_datetime(df['end_time'], format='%H:%M:%S')                
        except Exception:
            print("Error: it seems that some of the required columns of the frequencies.txt file are missing. ")    
        else:
            self.frequencies = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_trips(self):
        """ Setter for trips attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Trips: ", end="", flush=True)

        file_path = open(self.datapath / "trips.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'service_id', 'trip_id', 'shape_id']
        column_types = {'route_id': str, 'service_id': str, 'trip_id': str, 'shape_id': str}            

        try:          
            df = pd.read_csv(file_path, dtype=column_types)               
        except Exception:
            print("Error: it seems that some of the required columns of the trips.txt file are missing. ")    
        else:
            self.trips = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_shapes(self):
        """ Setter for shapes attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        print("\t Shapes: ", end="", flush=True)

        file_path = open(self.datapath / "shapes.txt", "r", encoding = "utf-8")       
       
        columns_to_keep = ['shape_id', 'shape_pt_lat', 'shape_pt_lon']                      

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep)
            # Group by shape_id and create LineString for each group
            grouped = df.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
            # Create a GeoDataFrame from the grouped data
            gdf = gpd.GeoDataFrame(geometry=grouped, crs="EPSG:4326").reset_index()                
        except Exception:
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the shapes.txt file are missing.")    
        else:
            self.shapes = gdf
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_stops(self):
        """ Setter for stops attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        print("\t Stops: ", end="", flush=True)

        file_path = open(self.datapath / "stops.txt", "r", encoding = "utf-8")       
    
        columns_to_keep = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']                     

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep)
            # Create Point geometries from latitude and longitude columns
            geometry = [Point(lon, lat) for lon, lat in zip(df['stop_lon'], df['stop_lat'])]
            # Drop latitude and longitude columns
            df = df.drop(['stop_lon', 'stop_lat'], axis=1)
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        except Exception:
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the stops.txt file are missing.")    
        else:
            self.stops = gdf
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close() 
