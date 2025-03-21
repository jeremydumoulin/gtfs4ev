# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shapely.ops import transform
import pyproj
from pyproj import Geod
from shapely.geometry import LineString, Point
import folium
from folium.plugins import TimestampedGeoJson

from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev import helpers as hlp

class TripSimulator:
    """Simulates the operation of a bus trip based on GTFS data."""

    def __init__(self, gtfs_manager: GTFSManager, trip_id: str):
        """
        Initializes the TripSimulator with GTFS data and a specific trip.

        Parameters:
        - gtfs_manager (GTFSManager): An instance of GTFSManager holding GTFS data.
        - trip_id (str): The trip ID to be simulated.
        """
        self.gtfs_manager = gtfs_manager
        self.trip_id = trip_id

        self._fleet_operation = None

    # Properties and Setters

    @property
    def gtfs_manager(self) -> GTFSManager:
        return self._gtfs_manager

    @gtfs_manager.setter
    def gtfs_manager(self, value):
        if not isinstance(value, object):  # Ideally, check against GTFSManager class
            raise ValueError("gtfs_manager must be a valid GTFSManager instance.")
        self._gtfs_manager = value

    @property
    def trip_id(self) -> str:
        return self._trip_id

    @trip_id.setter
    def trip_id(self, value):
        if self.gtfs_manager is None:
            raise ValueError("gtfs_manager must be set before setting trip_id.")
        if value not in self.gtfs_manager.trips['trip_id'].values:
            raise ValueError(f"Trip ID {value} not found in GTFS data.")
        self._trip_id = value

    # Number of vehicles in operation

    def trip_duration_sec(self) -> int:
        """Returns the duration of the trip in seconds."""
        trip_stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id]
        duration = (trip_stop_times['arrival_time'].iloc[-1] - trip_stop_times['departure_time'].iloc[0]).total_seconds()

        return duration

    def max_vehicles_in_operation(self) -> int:
        """
        Estimates the number of buses required to maintain the highest frequency in a given trip.

        Returns:
        - int: Estimated number of buses needed.
        """
        # Get the frequencies for this trip
        trip_frequencies = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]

        if trip_frequencies.empty:
            raise ValueError(f"No frequency data available for trip ID {self.trip_id}.")

        # Find the highest frequency (lowest headway)
        min_headway = trip_frequencies['headway_secs'].min()

        # Compute the number of vehicles needed
        max_num_vehicles = self.trip_duration_sec() / min_headway

        return max(1, round(max_num_vehicles))

    # Operation of the vehicles along the trip

    def simulate_fleet_operation(self, time_step: int) -> pd.DataFrame:
        """
        Simulates the operation of a fleet of vehicles with a time shift applied based on headway.
        The first vehicle's position remains unchanged, the second is shifted by one headway, the third
        by two headways, etc. This results in vehicles being further along the trip at the start of the
        frequency interval.

        Parameters:
          - time_step (int): Time step in seconds.

        Returns:
          - pd.DataFrame: DataFrame with each row as a vehicle and columns as time steps, 
            where each cell contains a dictionary with keys "status", "current_location", 
            and "incremental_distance" (from get_vehicle_status).
        """
        num_vehicles = self.max_vehicles_in_operation()
        time_range = pd.date_range("00:00:00", "23:59:59", freq=f"{time_step}S").time

        # Initialize dataframe with object dtype.
        df = pd.DataFrame(None, index=range(num_vehicles), columns=time_range, dtype=object)
        
        # Process each frequency interval for the current trip.
        trip_frequencies = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]
        for _, row in trip_frequencies.iterrows():
            start_time = row['start_time'].time()
            end_time = row['end_time'].time()
            headway = row['headway_secs']
            trip_duration = self.trip_duration_sec()
            
            # Estimate the number of vehicles in operation for this frequency interval.
            vehicles_in_operation = max(1, round(trip_duration / headway))
            print(f"Interval {start_time} to {end_time}: {vehicles_in_operation} vehicles in operation")
            
            # Randomly select which vehicles will operate in this interval.
            active_vehicles = np.random.choice(num_vehicles, vehicles_in_operation, replace=False)
            print(f"Selected vehicles: {active_vehicles}")
            
            # For the entire frequency interval, simulate the cyclic vehicle movement.
            current_time = start_time
            while current_time <= end_time:
                current_dt = datetime.combine(datetime.min, current_time)
                start_dt = datetime.combine(datetime.min, start_time)
                elapsed = (current_dt - start_dt).total_seconds()
                
                # For each active vehicle, shift its trip progress based on its order.
                for shift_idx, veh in enumerate(active_vehicles):
                    # Shift the elapsed time by shift_idx * headway.
                    shifted_elapsed = elapsed + shift_idx * headway
                    # Use modulo to repeat the trip once the terminal is reached.
                    duration_from_start = int(shifted_elapsed % trip_duration)
                    
                    # Get the vehicle status for the current elapsed time (within one trip cycle).
                    status_dict = self.get_vehicle_status(duration_from_start, time_step)
                    
                    # Assign the status to the vehicle at this time step.
                    if current_time in df.columns:
                        df.at[veh, current_time] = status_dict
                
                # Advance to the next time step.
                current_time = (datetime.combine(datetime.min, current_time) + timedelta(seconds=time_step)).time()
        
        self._fleet_operation = df

    def get_vehicle_status(self, duration_from_start: int, time_step: int) -> dict:
        """
        Determines the vehicle's status and computes its current location on the curved trip_linestring,
        then calculates the incremental distance traveled during the last time step (in meters).

        Parameters:
          - duration_from_start (int): Seconds from the trip's start.
          - time_step (int): The time step (in seconds) used to measure movement.

        Returns:
          dict: {
                "status": "at_stop", "at_terminal", or "travelling" (or np.nan if out of range),
                "current_location": A shapely Point on the trip's linestring,
                "incremental_distance": The geodesic distance (in meters) traveled between
                                        (duration_from_start - time_step) and duration_from_start.
               }
        """

        # Helper: compute status and location at a given time offset.
        def compute_status_at(duration: int):
            geod = Geod(ellps="WGS84")
            # Get stop times for the trip and sort them.
            stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id].copy()
            stop_times = stop_times.sort_values(by='arrival_time')
            first_stop_time = stop_times.iloc[0]['arrival_time']
            stop_times['arrival_offset'] = (stop_times['arrival_time'] - first_stop_time).dt.total_seconds()
            stop_times['departure_offset'] = (stop_times['departure_time'] - first_stop_time).dt.total_seconds()
            trip_linestring = self.gtfs_manager.get_shape(self.trip_id)

            # CASE 1: At a stop or terminal.
            for i, row in stop_times.iterrows():
                if row['arrival_offset'] <= duration <= row['departure_offset']:
                    stop_geom = self.gtfs_manager.stops.loc[
                        self.gtfs_manager.stops['stop_id'] == row['stop_id'], 'geometry'
                    ].iloc[0]
                    # Snap the stop to the line.
                    current_location = hlp.find_closest_point(trip_linestring, stop_geom)
                    status = "at_terminal" if (i == 0 or i == len(stop_times) - 1) else "at_stop"
                    return status, current_location

            # CASE 2: Outside scheduled stops.
            if duration < 0 or duration > stop_times.iloc[-1]['departure_offset']:
                return np.nan, None

            # CASE 3: Travelling between stops.
            previous_row = None
            next_row = None
            for idx in range(len(stop_times) - 1):
                dep_offset = stop_times.iloc[idx]['departure_offset']
                arr_offset_next = stop_times.iloc[idx + 1]['arrival_offset']
                if dep_offset < duration < arr_offset_next:
                    previous_row = stop_times.iloc[idx]
                    next_row = stop_times.iloc[idx + 1]
                    break

            if previous_row is None or next_row is None:
                return "travelling", None

            # Retrieve the geometries for the previous and next stops.
            prev_stop_geom = self.gtfs_manager.stops.loc[
                self.gtfs_manager.stops['stop_id'] == previous_row['stop_id'], 'geometry'
            ].iloc[0]
            next_stop_geom = self.gtfs_manager.stops.loc[
                self.gtfs_manager.stops['stop_id'] == next_row['stop_id'], 'geometry'
            ].iloc[0]

            # Snap these stop points to the trip_linestring.
            prev_point = hlp.find_closest_point(trip_linestring, prev_stop_geom)
            next_point = hlp.find_closest_point(trip_linestring, next_stop_geom)

            # Use linear referencing to get positions along the curved trip_linestring.
            dist_prev = trip_linestring.project(prev_point)
            dist_next = trip_linestring.project(next_point)
            
            # Calculate the time fraction between the two stops.
            time_segment = next_row['arrival_offset'] - previous_row['departure_offset']
            time_elapsed = duration - previous_row['departure_offset']
            fraction = time_elapsed / time_segment if time_segment > 0 else 0
            
            # Interpolate along the curved line.
            interp_distance = dist_prev + fraction * (dist_next - dist_prev)
            current_location = trip_linestring.interpolate(interp_distance)
            return "travelling", current_location

        # Compute current and previous positions.
        status_current, current_location = compute_status_at(duration_from_start)
        if duration_from_start - time_step >= 0:
            _, prev_location = compute_status_at(duration_from_start - time_step)
        else:
            prev_location = current_location

        geod = Geod(ellps="WGS84")
        if current_location is not None and prev_location is not None:
            # Compute incremental distance between the two points.
            _, _, incremental_distance = geod.inv(prev_location.x, prev_location.y,
                                                   current_location.x, current_location.y)
        else:
            incremental_distance = None

        return {
            "status": status_current,
            "current_location": current_location,
            "incremental_distance": incremental_distance
        }

    # Export and visualisation


    def create_map_with_slider(self):
        """
        Generates an interactive folium map with a time slider using the simulated fleet operation data.

        Warning: minimum time step is 2 minutes !

        """
        if self._fleet_operation is None:
            raise ValueError("Fleet operation data is not available. Run simulate_fleet_operation() first.")
        
        # Transform the fleet operation DataFrame into a long-format DataFrame
        data = []
        for vehicle_id in self._fleet_operation.index:
            for time, status in self._fleet_operation.loc[vehicle_id].items():
                if not isinstance(status, dict) or status.get("current_location") is None:
                    continue  # Skip invalid data
                
                data.append({
                    "vehicle_id": vehicle_id,
                    "time": datetime.combine(datetime.today(), time),
                    "latitude": status["current_location"].y,
                    "longitude": status["current_location"].x
                })

        # Convert to DataFrame and check validity
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("No valid location data to plot.")
        
        # Sort and clean data
        df.sort_values(by=["time", "vehicle_id"], inplace=True)
        df.drop_duplicates(subset=["time", "vehicle_id"], inplace=True)

        # Calculate time step in minutes
        unique_times = df["time"].drop_duplicates().sort_values().reset_index(drop=True)
        if len(unique_times) < 2:
            raise ValueError("Not enough timestamps to determine time step.")

        time_diffs = unique_times.diff().dropna()
        min_step = int(min(time_diffs).total_seconds() // 60)  # in minutes

        # Build ISO 8601 strings
        period_str = f"PT{min_step}M"
        duration_str = f"PT{max(min_step - 1, 1)}M"  # Avoid PT0M

        # Create base map
        first_position = df.iloc[0]
        m = folium.Map(location=[first_position['latitude'], first_position['longitude']], zoom_start=12)

        # Add trip shape (polyline) to map
        trip_linestring = self.gtfs_manager.get_shape(self.trip_id)

        if trip_linestring and not trip_linestring.is_empty:
            if hasattr(trip_linestring, "coords"):
                shape_coords = [(lat, lon) for lon, lat in trip_linestring.coords]  # Folium expects (lat, lon)
                folium.PolyLine(
                    shape_coords,
                    color="black",
                    weight=4,
                    opacity=0.7,
                    tooltip="Trip Shape"
                ).add_to(m)
        
        # Organize data into time-based FeatureCollections
        time_grouped = df.groupby("time")
        features = []
        
        for timestamp, group in time_grouped:
            vehicle_points = []
            for _, row in group.iterrows():
                vehicle_points.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["longitude"], row["latitude"]]
                    },
                    "properties": {
                        "time": timestamp.isoformat(),
                        "popup": f"Vehicle {row['vehicle_id']}\nTime: {timestamp.strftime('%H:%M:%S')}"
                    }
                })
            
            features.append({
                "type": "FeatureCollection",
                "features": vehicle_points
            })

        # Add dynamic time slider
        TimestampedGeoJson({
            "type": "FeatureCollection",
            "features": features
        },
        period=period_str,
        duration=duration_str,
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True).add_to(m)

        return m

