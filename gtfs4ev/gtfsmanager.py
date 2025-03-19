# coding: utf-8

"""
GTFSFeed
--------

"""

import os

import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, Point, box
from shapely.ops import transform, snap
import pyproj
from pyproj import Geod
import osmnx
from contextlib import redirect_stdout
import folium
from folium.plugins import MarkerCluster

from gtfs4ev import helpers as hlp

class GTFSManager:
    """A class to represent and manage a GTFS feed data.

    Holds and curates the GTFS feed. This class reads a GTFS data folder and provides methods
    to check data consistency, filter and clean data, and extract general transit indicators.

    Note: The calendar_dates.txt file is not considered, so some service exceptions are not handled.
    """

    def __init__(self, gtfs_datafolder):
        print("=========================================")
        print(f"INFO \t Creation of a GTFSManager object.")
        print("=========================================")

        self._gtfs_datafolder = None
        self._agency = pd.DataFrame()
        self._routes = pd.DataFrame()
        self._stop_times = pd.DataFrame()
        self._stops = gpd.GeoDataFrame(columns=['stop_id', 'stop_name', 'geometry'], crs="EPSG:4326")
        self._trips = pd.DataFrame()
        self._calendar = pd.DataFrame()
        self._frequencies = pd.DataFrame()
        self._shapes = gpd.GeoDataFrame(columns=['shape_id', 'geometry'], crs="EPSG:4326")
       
        self.gtfs_datafolder = gtfs_datafolder

        # Load the datasets using the helper method
        self._agency = self.load_csv("agency.txt", 
                                      columns=['agency_id', 'agency_name', 'agency_url', 'agency_timezone'],
                                      dtypes={col: str for col in ['agency_id', 'agency_name', 'agency_url', 'agency_timezone']})
        self._trips = self.load_csv("trips.txt",
                                     columns=['route_id', 'service_id', 'trip_id', 'shape_id'],
                                     dtypes={col: str for col in ['route_id', 'service_id', 'trip_id', 'shape_id']})
        self._routes = self.load_csv("routes.txt", 
                                      columns=['route_id', 'agency_id', 'route_short_name', 'route_long_name'],
                                      dtypes={col: str for col in ['route_id', 'agency_id', 'route_short_name', 'route_long_name']})
        self._stop_times = self.load_csv("stop_times.txt",
                                          columns=['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence'],
                                          dtypes={'trip_id': str, 'arrival_time': str, 'departure_time': str, 'stop_id': str, 'stop_sequence': int},
                                          parse_dates={'arrival_time': '%H:%M:%S', 'departure_time': '%H:%M:%S'})
        self._calendar = self.load_csv("calendar.txt",
                                        dtypes={'service_id': str, 'monday': bool, 'tuesday': bool, 'wednesday': bool,
                                                'thursday': bool, 'friday': bool, 'saturday': bool, 'sunday': bool,
                                                'start_date': str, 'end_date': str},
                                        parse_dates={'start_date': '%Y%m%d', 'end_date': '%Y%m%d'})
        self._frequencies = self.load_csv("frequencies.txt",
                                           columns=['trip_id', 'start_time', 'end_time', 'headway_secs'],
                                           dtypes={'trip_id': str, 'start_time': str, 'end_time': str, 'headway_secs': int},
                                           parse_dates={'start_time': '%H:%M:%S', 'end_time': '%H:%M:%S'},
                                           replace_times={'24:00:00': '23:59:59'})
        self._stops = self.load_stops()
        self._shapes = self.load_shapes()

        print("INFO \t Successful initialization of the GTFSManager. The GTFS feed could now be analyzed, cleaned, or filtered.")

    # Properties and Setters

    @property
    def gtfs_datafolder(self) -> str:
        """str: The absolute path to the GTFS data folder as a string."""
        if self._gtfs_datafolder is None:
            raise ValueError("GTFS data folder has not been set.")
        return self._gtfs_datafolder

    @gtfs_datafolder.setter
    def gtfs_datafolder(self, abs_path):
        abs_path = Path(abs_path)  # Convert string to Path object
        required_files = ['agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt',
                          'frequencies.txt', 'shapes.txt', 'stops.txt', 'trips.txt']
        if not abs_path.is_dir():
            raise FileNotFoundError(f"ERROR \t Unable to find folder: /{gtfs_foldername}")
        for file_name in required_files:
            if not (abs_path / file_name).exists():
                raise FileNotFoundError(f"ERROR \t Required file '{file_name}' is missing in folder /{gtfs_foldername}")
        for file_name in os.listdir(abs_path):
            if file_name not in required_files:
                print(f"INFO \t The data folder contains a GTFS file that is not required: '{file_name}'.")
        self._gtfs_datafolder = abs_path

    @property
    def agency(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS agency data."""
        return self._agency

    @property
    def routes(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS routes data."""
        return self._routes

    @property
    def stop_times(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS stop times data."""
        return self._stop_times

    @property
    def stops(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS stops data, including geometries."""
        return self._stops

    @property
    def trips(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS trips data."""
        return self._trips

    @property
    def calendar(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS calendar data."""
        return self._calendar

    @property
    def frequencies(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS frequencies data."""
        return self._frequencies

    @property
    def shapes(self) -> pd.DataFrame:
        """pd.DataFrame: The GTFS shapes data as a GeoDataFrame containing LineString geometries."""
        return self._shapes

    # Data validation

    def check_agency(self):
        """ Checking that each agency is associated with at least one route
        """
        problem = False
        agency_ids = self.agency['agency_id']
        
        all_present = agency_ids.isin(self.routes['agency_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some agencies are not associated with any route.")

        return problem 

    def check_shapes(self):
        """ Checking that each shape is associated with at least one trip
        """
        problem = False

        shape_ids = self.shapes['shape_id']
        
        all_present = shape_ids.isin(self.trips['shape_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some shapes are not associated with any trip.")

        return problem

    def check_stops(self):
        """ Checking that each stop is associated with at least one stop time
        """
        problem = False

        stop_ids = self.stops['stop_id']
        
        all_present = stop_ids.isin(self.stop_times['stop_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stops are not associated with any stop time.")

        return problem

    def check_frequencies(self):
        """ Checking that each frequency is associated with at least one stop trip
        """
        problem = False

        trip_ids = self.frequencies['trip_id']
        
        all_present = trip_ids.isin(self.trips['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some frequencies are not associated with any trip.")

        return problem

    def check_calendar(self):
        """ Checking that each service is associated with at least one stop trip
        """
        problem = False

        service_ids = self.calendar['service_id']
        
        all_present = service_ids.isin(self.trips['service_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some services are not associated with any trip.")

        return problem

    def check_routes(self):
        """ Checking that each route is associated with an agency and trips
        """
        problem = False

        route_ids = self.routes['route_id']
        agency_ids = self.routes['agency_id']
        
        all_present = route_ids.isin(self.trips['route_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some routes are not associated with any trip.")

        all_present = agency_ids.isin(self.agency['agency_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some routes are not associated with any agency.")

        return problem

    def check_stop_times(self):
        """ Checking that each stop time is associated with trips and stops
        """
        problem = False

        stop_ids = self.stop_times['stop_id']
        trip_ids = self.stop_times['trip_id']
        
        all_present = stop_ids.isin(self.stops['stop_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stop times are not associated with any stop.")

        all_present = trip_ids.isin(self.trips['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stop times are not associated with any trip.")

        return problem

    def check_trips(self):
        """ Checking that each trips is associated with the needed information: routes, service, frequencies, stop_times, shapes
        """
        problem = False

        route_ids = self.trips['route_id']
        service_ids = self.trips['service_id']
        trip_ids = self.trips['trip_id']
        shape_ids = self.trips['shape_id']
        
        all_present = route_ids.isin(self.routes['route_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any route.")

        all_present = service_ids.isin(self.calendar['service_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any agency.")

        all_present = trip_ids.isin(self.frequencies['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any frequencies. Could cause important issues.")

        all_present = trip_ids.isin(self.stop_times['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any stop times.")  

        all_present = shape_ids.isin(self.shapes['shape_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any shape.")

        return problem

    def check_all(self): 
        """ Checking the consistency of all the data
        """
        print("INFO \t Checking data consistency.")

        consistent = False

        agency_status = self.check_agency()
        shapes_status = self.check_shapes()
        stops_status = self.check_stops()
        frequencies_status = self.check_frequencies()
        calendar_status = self.check_calendar()
        routes_status = self.check_routes()
        stop_times_status = self.check_stop_times()
        trips_status = self.check_trips()

        if (not agency_status and not shapes_status and not stops_status and not frequencies_status and not calendar_status and not routes_status and not stop_times_status and not trips_status):
            print("\t No problems found.")
            consistent = True

        return consistent

    # Data cleaning 

    def clean_agency(self):
        """Deletes each agency not associated with any route."""
        self._agency = self._agency[self._agency['agency_id'].isin(self._routes['agency_id'])]

    def clean_shapes(self):
        """Deletes each shape not associated with any trip."""
        self._shapes = self._shapes[self._shapes['shape_id'].isin(self._trips['shape_id'])]

    def clean_stops(self):
        """Deletes each stop not associated with any stop_time."""
        self._stops = self._stops[self._stops['stop_id'].isin(self._stop_times['stop_id'])]

    def clean_frequencies(self):
        """Deletes each frequency not associated with any trip."""
        self._frequencies = self._frequencies[self._frequencies['trip_id'].isin(self._trips['trip_id'])]

    def clean_calendar(self):
        """Deletes each service not associated with any trip."""
        self._calendar = self._calendar[self._calendar['service_id'].isin(self._trips['service_id'])]

    def clean_routes(self):
        """Deletes each route not associated with any agency or trip."""
        self._routes = self._routes[self._routes['agency_id'].isin(self._agency['agency_id'])]
        self._routes = self._routes[self._routes['route_id'].isin(self._trips['route_id'])]

    def clean_stop_times(self):
        """Deletes each stop_time not associated with any trip or stop."""
        self.drop_useless_stop_times()
        self._stop_times = self._stop_times[self._stop_times['trip_id'].isin(self._trips['trip_id'])]

    def clean_trips(self):
        """Deletes each trip not associated with any route, service, frequency, stop_time, or shape."""
        self.drop_useless_trips()
        self._trips = self._trips[self._trips['route_id'].isin(self._routes['route_id'])]
        self._trips = self._trips[self._trips['service_id'].isin(self._calendar['service_id'])]

    def drop_useless_trips(self):
        """ Delete the trips which are not useable because they are not assiocated to any frequency, stop_times, or shapes
        Warning: This ensures that all the trips could be simulated by gtfs4ev but could alter data consistency: do a cleaning to ensure th whole dataset is consistent
        """
        trips = self.trips
        frequencies = self.frequencies
        stop_times = self.stop_times
        shapes = self.shapes

        # Drop trips not associated to any frequency
        merged_df = trips.merge(frequencies, on='trip_id')        
        cleaned_trips = trips[trips['trip_id'].isin(merged_df['trip_id'])] # Only keep the useful rows

        # Drop trips not associated to any stop_times
        merged_df = cleaned_trips.merge(stop_times, on='trip_id')        
        cleaned_trips_2 = cleaned_trips[cleaned_trips['trip_id'].isin(merged_df['trip_id'])] # Only keep the useful rows

        # Drop trips not associated to any shape
        merged_df = cleaned_trips_2.merge(shapes, on='shape_id')        
        cleaned_trips_3 = cleaned_trips_2[cleaned_trips_2['shape_id'].isin(merged_df['shape_id'])] # Only keep the useful rows

        self._trips = cleaned_trips_3

    def drop_useless_stop_times(self):
        """ Delete the stop_times which are not useable because they are not assiocated to any stop
        Warning: This ensures that all the stop_times could be simulated by gtfs4ev but could alter data consistency: do a cleaning to ensure th whole dataset is consistent
        """        
        stop_times = self.stop_times
        stops = self.stops

        # Drop stop_times not associated to any frequency
        merged_df = stop_times.merge(stops, on='stop_id')        
        cleaned_stop_times = stop_times[stop_times['stop_id'].isin(merged_df['stop_id'])] # Only keep the useful rows

        self._stop_times = cleaned_stop_times

    def clean_all(self):
        """Executes all the cleaning functions to ensure a fully consistent and usable dataset."""

        # Clean everything and repeat until consistency is reached
        print("INFO \t Starting GTFS dataset cleaning to ensure a fully consistent and usable dataset...")
        print("\t - Removing unreferenced agencies, routes, trips, stops, stop_times, frequencies, and shapes.")
        print("\t - Ensuring all trips, stop_times, shapes, and frequencies are correctly linked.")

        with redirect_stdout(None): 
            while not self.check_all():
                self.clean_agency()
                self.clean_shapes()
                self.clean_stops()
                self.clean_frequencies()
                self.clean_calendar()
                self.clean_routes() 
                self.clean_stop_times()
                self.clean_trips()

        print("\t Data cleaning completed successfully. The dataset is now consistent and ready for simulation.")

    def snap_shapes_to_osm(self):
        """ Snapping the shapefile to the local OSM road network """ 
        print("INFO \t Snapping the GTFS trip shapes to OSM road network. This might take a long time...")
        bbox = self.bounding_box()
        print("INFO \t Getting OSM data according to GTFS data boundaries")
        graph = osmnx.graph_from_bbox(bbox = (bbox.bounds[3], bbox.bounds[1], bbox.bounds[2], bbox.bounds[0]), network_type='drive')

        # Get the nodes from the road network graph
        nodes = osmnx.graph_to_gdfs(graph, edges=False)

        # Snap each LineString geometry to the nearest node in the road network
        for idx, row in self.shapes.iterrows():
            print(f"INFO \t Snapping in progress: {idx}/{len(self.shapes)}", end="\r")
            snapped_line = snap(row.geometry, nodes.unary_union, tolerance=0.0001)
            self._shapes.loc[idx, 'geometry'] = snapped_line     

    # Data filtering

    def filter_services(self, service_id, clean_all = True):
        """ Drop the data belonging to a specific service, for example to consider only weekdays
        """

        print(f"INFO \t Filtering out all the data from the following service: {service_id}")

        # Drop the trips associated to a specific service       
        self._trips.drop(self.trips[self.trips['service_id'] == service_id].index, inplace=True)

        # Clean all the dataset to be consistent with the trips
        if clean_all:
            with redirect_stdout(None): 
                self.clean_all()

    def filter_agency(self, agency_id, clean_all = True):
        """ Drop the data belonging to a specific agency, for example to consider only paratransit
        """

        print(f"INFO \t Filtering out all the data from the following agency: {agency_id}")

        # Drop the routes associated to a specific agency       
        self._routes.drop(self.routes[self.routes['agency_id'] == agency_id].index, inplace=True)

        # Clean all the dataset to be consistent with the routes
        if clean_all:
            with redirect_stdout(None): 
                self.clean_all()

    # Data analysis

    """ Basic aggregated indicators """

    def general_feed_info(self):
        """Displays general information about the GTFS feed in a clean and readable format."""        

        print("INFO \t 🚍 GTFS Feed Summary:")

        print(f"\t {'Trips:':<15} {self.trips.shape[0]:>8}  |  {'Routes:':<15} {self.routes.shape[0]:>8}  |  {'Services:':<15} {self.calendar.shape[0]:>8}")
        print(f"\t {'Stops:':<15} {self.stops.shape[0]:>8}  |  {'Stop Times:':<15} {self.stop_times.shape[0]:>8}  |  {'Frequencies:':<15} {self.frequencies.shape[0]:>8}")
        print(f"\t {'Agencies:':<15} {self.agency.shape[0]:>8}  |  {'Shapes:':<15} {self.shapes.shape[0]:>8}")
        

        # Check if the number of trips is exactly twice the number of routes (potential round-trip pattern)
        if self.trips.shape[0] / self.routes.shape[0] == 2.0:
            print("\t🔹Note: The number of trips is twice the number of routes, suggesting that each route is associated with a round trip.")

        # Analyze frequency consistency
        group_sizes = self.frequencies.groupby('trip_id').size().reset_index(name='row_count')
        are_all_values_same = group_sizes['row_count'].nunique() == 1

        print(f"\t Simulation area: {self.simulation_area_km2()} km2")

        if are_all_values_same:
            print("\t Temporal Analysis:")
            print(f"\t  - Frequency intervals: {group_sizes['row_count'][0]}")
            print(f"\t  - Operating from {self.frequencies['start_time'].min()} to {self.frequencies['end_time'].max()}")
        else:
            print("\t🔹Note: The number of frequency intervals is inconsistent across trips.")


    """ Per trip transit indicators """

    def trip_length_km(self, trip_id, geodesic = True) -> float:
        """ Calculates the lenght in km of a trip.
        """
        # Get the shape of the corresponding trip and project it into epsg:3857 crs
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        if geodesic: 
            geod = Geod(ellps="WGS84")
            distance = geod.geometry_length(linestring) / 1000.0
        else:
            web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
            linestring_projection = transform(web_mercator_projection, linestring)
            distance = linestring_projection.length / 1000.0

        return distance

    def trip_duration_sec(self, trip_id) -> float:
        """ Calculates the time in sec of a trip
        """
        trip_stop_times = self.stop_times[self.stop_times['trip_id'] == trip_id]
        duration = (trip_stop_times['arrival_time'].iloc[-1] - trip_stop_times['departure_time'].iloc[0]).total_seconds()

        return duration

    def ave_distance_between_stops(self, trip_id, correct_stop_loc = True) -> float:
        """ Calculates the average distance between stops in km along a trip
        If needed, the stop location can by clipped to the closest point along the shape of the trip
        """
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 

        linestring_web_mercator_projection = transform(web_mercator_projection, linestring)

        # 2. Create a table which contains the stop times for the trip, the stop name and geometry
        filtered_stop_times = self.stop_times[self.stop_times['trip_id'] == trip_id]        
        result_df = pd.merge(filtered_stop_times, self.stops[['stop_id', 'stop_name', 'geometry']], on='stop_id', how='left') # Merge the filtered_df with the stop_points_df on 'stop_id'

        # 3. Add the closest point on the linestring path as the point for the stop
        if correct_stop_loc:
            result_df['closest_point_on_path'] = result_df['geometry'].apply(lambda point: hlp.find_closest_point(linestring, point))
        else:
            result_df['closest_point_on_path'] = result_df['geometry']

        result_df['stop_dist_km'] = .0

        # Define the projection from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857) 
        for i in range(1, len(result_df)):
            current_stop = result_df.iloc[i] 
            previous_stop = result_df.iloc[i - 1]

            start_point = transform(web_mercator_projection, previous_stop['closest_point_on_path'])
            end_point = transform(web_mercator_projection, current_stop['closest_point_on_path'])

            distance_along_line = abs(linestring_web_mercator_projection.project(start_point) - linestring_web_mercator_projection.project(end_point)) 

            result_df.loc[result_df.index == i, 'stop_dist_km'] = distance_along_line

        return result_df['stop_dist_km'].mean() / 1000.0

    def n_stops(self, trip_id) -> int:
        """ Number of stops of a trip
        """        
        stop_times = self.stop_times       
        n_stops = stop_times.groupby('trip_id').size().reset_index(name='row_count')

        return n_stops.loc[n_stops['trip_id'] == trip_id, 'row_count'].iloc[0]

    """ Per stop transit indicators """

    def stop_frequencies(self) -> int:
        """ Returns the number of times a stop is used by all trips
        """
        stop_times = self.stop_times

        stop_counts = stop_times.groupby('stop_id').size().reset_index(name='count')
        stop_counts = pd.merge(stop_counts, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')
        
        return stop_counts

    """ Global transit indicators """

    def bounding_box(self) -> box:
        """ Returns the bounding box for the simulation    
        """
        gdf = self.shapes

        self.min_latitude = gdf['geometry'].apply(lambda line: line.bounds[1]).min()
        self.max_latitude = gdf['geometry'].apply(lambda line: line.bounds[3]).max()
        self.min_longitude = gdf['geometry'].apply(lambda line: line.bounds[0]).min()
        self.max_longitude = gdf['geometry'].apply(lambda line: line.bounds[2]).max()

        return box(self.min_longitude, self.min_latitude, self.max_longitude, self.max_latitude)

    def simulation_area_km2(self) -> float:
        """ Calculates the simulation area in km2
        """
        # Create a GeoDataFrame with the bounding box
        gdf_bbox = gpd.GeoDataFrame(geometry=[self.bounding_box()], crs='epsg:4326')

        # Reproject the GeoDataFrame to EPSG:3857
        gdf_bbox_web_mercator = gdf_bbox.to_crs('epsg:3857')

        # Calculate the area in square kilometers
        area_km2 = gdf_bbox_web_mercator['geometry'].area / 1e6  # Convert square meters to square kilometers

        return area_km2[0]  

    def trip_length_km_all(self) -> float:
        """ Length of all trips
        """        
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')

        gdf['trip_length_km'] = gdf['trip_id'].apply(self.trip_length_km)

        return gdf       

    def ave_distance_between_stops_all(self, correct_stop_loc = True) -> float:
        """ Average distance between stops in km of all trips
        """        
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')

        gdf['stop_dist_km'] = gdf['trip_id'].apply(lambda trip_id: self.ave_distance_between_stops(trip_id, correct_stop_loc=correct_stop_loc))
        gdf['n_stops'] = gdf['trip_id'].apply(self.n_stops)

        return gdf

    def trip_statistics(self) -> dict:
        """ Calculates some general statitistics regarding the trips
        """   
        number_of_trips = len(self.trips)
        number_of_routes = len(self.routes)
        gdf = self.trip_length_km_all()

        # Calculate the minimum, maximum, average, and standard deviation of the number of rows per trip_id
        statistics = {
            'total_trips': number_of_trips,
            'total_trip_len_km': gdf['trip_length_km'].sum(),
            'ave_trip_len_km': gdf['trip_length_km'].mean(),
            'min_trip_len_km': gdf['trip_length_km'].min(),
            'max_trip_len_km': gdf['trip_length_km'].max(), 
            'trip_to_route_ratio': number_of_trips / number_of_routes
        }
        
        return statistics

    def stop_statistics(self) -> dict:
        """ Calculates some general statistics regarding the number of stops
        """
        # Add the route information to the stop_times by merging the two DataFrames on 'trip_id'
        stop_times = pd.merge(self.stop_times, self.trips[['trip_id', 'route_id']], on='trip_id', how='left')

        number_of_stops = len(self.stops)
        number_of_trips = len(self.trips)
        number_of_routes = len(self.routes)

        # Calculate the minimum, maximum, average, and standard deviation of the number of rows per trip_id
        statistics = {
            'total_stops': number_of_stops,
            'min_stops_per_trip': stop_times.groupby('trip_id').size().min(),
            'max_stops_per_trip': stop_times.groupby('trip_id').size().max(),
            'std_dev_stops_per_trip': stop_times.groupby('trip_id').size().std(),
            'ave_stops_per_trip': stop_times.groupby('trip_id').size().mean(),
            'ave_stops_per_route': stop_times.groupby('route_id').size().mean(),
            'stops_to_trips_ratio': number_of_stops / number_of_trips,
            'stops_to_routes_ratio': number_of_stops / number_of_routes            
        }
        
        return statistics  

    # Helper methods

    def load_csv(self, filename, columns=None, dtypes=None, parse_dates=None, replace_times=None):
        """
        Generic helper method to load a CSV file with optional column selection,
        data type conversion, and date parsing.
        """
        file_path = self.gtfs_datafolder / filename
        try:
            df = pd.read_csv(file_path, usecols=columns, dtype=dtypes)
            if replace_times:
                for col in ['start_time', 'end_time']:
                    if col in df.columns:
                        df[col] = df[col].str.replace('24:00:00', replace_times.get('24:00:00', '23:59:59'))
            if parse_dates:
                for col, fmt in parse_dates.items():
                    df[col] = pd.to_datetime(df[col], format=fmt)
        except Exception as e:
            raise ValueError(f"ERROR \t Problem loading {filename}.") from e
        return df

    def load_stops(self):
        """Loads and returns the stops GeoDataFrame."""
        file_path = self.gtfs_datafolder / "stops.txt"
        try:
            df = pd.read_csv(file_path, usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'],
                             dtype={'stop_id': str, 'stop_name': str, 'stop_lat': float, 'stop_lon': float})
            geometry = [Point(lon, lat) for lon, lat in zip(df['stop_lon'], df['stop_lat'])]
            df.drop(['stop_lon', 'stop_lat'], axis=1, inplace=True)
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        except Exception as e:
            raise ValueError("ERROR \t Problem creating GeoDataFrame from stops.txt.") from e
        return gdf

    def load_shapes(self):
        """Loads and returns the shapes GeoDataFrame as LineStrings."""
        file_path = self.gtfs_datafolder / "shapes.txt"
        try:
            df = pd.read_csv(file_path, usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon'],
                             dtype={'shape_id': str, 'shape_pt_lat': float, 'shape_pt_lon': float})
            grouped = df.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
            gdf = gpd.GeoDataFrame(geometry=grouped, crs="EPSG:4326").reset_index()
        except Exception as e:
            raise ValueError("ERROR \t Problem creating GeoDataFrame from shapes.txt.") from e
        return gdf

    def get_shape(self, trip_id):
        """ Get the shape of a trip_id as a LineString Object
        """
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        return linestring

    def get_stop_locations(self, trip_id):
        """ Get a list of the coordinates of all the stops belonging to a trip 
        """
        gdf = pd.merge(self.stop_times, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')       
        coordinates = gdf.loc[gdf['trip_id'] == trip_id, 'geometry']

        return coordinates.tolist()

    # Export and visualisation

    def to_map(self, filepath: str) -> None:
        """
        Visualizes the different routes on a Folium map and saves it to the specified file path.
        
        Parameters:
        filepath (str): The path where the map will be saved as an HTML file.
        """
        print("INFO \t Generating the GTFS map visualization. This may take some time...")
        
        # Get the bounding box center
        bbox = self.bounding_box()
        center_lat = (bbox.bounds[1] + bbox.bounds[3]) / 2
        center_lon = (bbox.bounds[0] + bbox.bounds[2]) / 2
        
        # Initialize map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add route polylines with popups for trip details
        for _, row in self.shapes.iterrows():
            related_trips = self.trips[self.trips['shape_id'] == row['shape_id']]
            trip_info = ""
            for _, trip in related_trips.iterrows():
                trip_id = trip['trip_id']
                trip_length = self.trip_length_km(trip_id)
                num_stops = self.n_stops(trip_id)
                trip_info += f"<b>Trip ID:</b> {trip_id}<br><b>Length:</b> {trip_length:.2f} km<br><b>Stops:</b> {num_stops}<br><br>"
            popup_content = f"<b>Shape ID:</b> {row['shape_id']}<br>{trip_info if trip_info else 'No trip information'}"
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in row['geometry'].coords],
                color='blue',
                weight=3,
                opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)
        
        # Add stop markers
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in self.stops.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=f"Stop: {row['stop_name']} ({row['stop_id']})",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        
        # Save the map
        m.save(filepath)
        print(f"INFO \t Map successfully generated and saved to {filepath}")

    def export_statistics(self, filepath: str) -> None:
        """
        Exports key GTFS statistics and results to a text file.
        
        Parameters:
        filepath (str): The path where the statistics will be saved.
        """
        print("INFO \t Exporting GTFS statistics to a text file...")
        
        trip_stats = self.trip_statistics()
        stop_stats = self.stop_statistics()
        area_km2 = self.simulation_area_km2()
        
        with open(filepath, 'w') as f:
            f.write("GTFS Feed Summary:\n")
            f.write("===========================================\n\n")
            f.write(f"Trips: {trip_stats['total_trips']}  |  Routes: {len(self.routes)}  |  Services: {len(self.calendar)}\n")
            f.write(f"Stops: {stop_stats['total_stops']}  |  Stop Times: {len(self.stop_times)}  |  Frequencies: {len(self.frequencies)}\n")
            f.write(f"Agencies: {len(self.agency)}  |  Shapes: {len(self.shapes)}\n\n")
            
            if trip_stats['trip_to_route_ratio'] == 2.0:
                f.write("Note: The number of trips is twice the number of routes, suggesting that each route is associated with a round trip.\n\n")
            
            f.write(f"Simulation area: {area_km2:.2f} km²\n\n")

            f.write("===========================================\n\n")
            
            f.write("Temporal Analysis:\n")
            f.write("=================\n\n")
            
            # Analyze frequency consistency
            group_sizes = self.frequencies.groupby('trip_id').size().reset_index(name='row_count')
            are_all_values_same = group_sizes['row_count'].nunique() == 1

            if are_all_values_same:
                f.write(f"Frequency intervals: {group_sizes['row_count'][0]} \n")
                f.write(f"Operating from {self.frequencies['start_time'].min()} to {self.frequencies['end_time'].max()}")
            else:
                f.write("\t Note: The number of frequency intervals is inconsistent across trips.")
            
            f.write("\n\nTrip Statistics:\n")
            f.write("=================\n\n")
            f.write(f"Total Trip Length (km): {trip_stats['total_trip_len_km']:.2f}\n")
            f.write(f"Average Trip Length (km): {trip_stats['ave_trip_len_km']:.2f}\n")
            f.write(f"Min Trip Length (km): {trip_stats['min_trip_len_km']:.2f}\n")
            f.write(f"Max Trip Length (km): {trip_stats['max_trip_len_km']:.2f}\n\n")
            
            f.write("Stop Statistics:\n")
            f.write("=================\n\n")
            f.write(f"Min Stops per Trip: {stop_stats['min_stops_per_trip']}\n")
            f.write(f"Max Stops per Trip: {stop_stats['max_stops_per_trip']}\n")
            f.write(f"Avg Stops per Trip: {stop_stats['ave_stops_per_trip']:.2f}\n")
            f.write(f"Avg Stops per Route: {stop_stats['ave_stops_per_route']:.2f}\n\n")
            
        print(f"INFO \t Statistics successfully exported to {filepath}")
