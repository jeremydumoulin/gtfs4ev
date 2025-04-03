# coding: utf-8

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from shapely.ops import substring
from shapely.geometry import LineString, Point
import folium
from folium.plugins import TimestampedGeoJson

from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev import helpers as hlp

class TripSimulator:
    """A class simulating the operation of vehicles on a trip based on GTFS data."""

    def __init__(self, gtfs_manager: GTFSManager, trip_id: str):
        """
        Initializes the TripSimulator class.

        Args:
            gtfs_manager (GTFSManager): An instance of GTFSManager holding GTFS data.
            trip_id (str): The trip ID to be simulated.
        """
        self.gtfs_manager = gtfs_manager
        self.trip_id = trip_id

        self._single_trip_sequence = None
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

    @property
    def single_trip_sequence(self) -> pd.DataFrame:
        """pd.DataFrame: The sequence of events of a single vehicle along the trip."""
        return self._single_trip_sequence

    @property
    def fleet_operation(self) -> pd.DataFrame:
        """pd.DataFrame: The operation schedule of all vehicles."""
        return self._fleet_operation

    # Fleet operation

    def compute_single_trip_sequence(self):
        """
        Computes the travel sequence for a single trip based on GTFS data.

        This function determines the sequence of events for a given trip, including stops,
        travel distances, and durations. It returns a structured list of events, where each 
        event contains:

        - status: "at_terminal", "at_stop", or "travelling"
        - distance: distance traveled (km) during "travelling"; 0 for stops
        - duration: event duration in seconds
        - distance_from_start: cumulative distance from trip start
        - duration_from_start: cumulative duration from trip start
        - location: a Point (lat, lon) for stops and terminals, or a LineString for travelling.

        Returns:
            List[Dict]: A structured list of trip events.
        """

        # --- Step 1: Extract and sort stop times for the trip ---
        stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id].copy()
        stop_times.sort_values(by="arrival_time", inplace=True)

        # Compute time offsets relative to the first arrival
        first_arrival = stop_times.iloc[0]["arrival_time"]
        stop_times["arrival_offset"] = (stop_times["arrival_time"] - first_arrival).dt.total_seconds()
        stop_times["departure_offset"] = (stop_times["departure_time"] - first_arrival).dt.total_seconds()

        # --- Step 2: Retrieve stop locations as a dictionary {stop_id: Point(lat, lon)} ---
        stop_geometries = {
            row['stop_id']: Point(
                self.gtfs_manager.stops.loc[self.gtfs_manager.stops['stop_id'] == row['stop_id'], 'geometry'].iloc[0].x,
                self.gtfs_manager.stops.loc[self.gtfs_manager.stops['stop_id'] == row['stop_id'], 'geometry'].iloc[0].y
            )
            for _, row in stop_times.iterrows()
        }

        # --- Step 3: Retrieve trip shape and compute distances to stops ---
        trip_shape = self.gtfs_manager.get_shape(self.trip_id)

        # Compute stop distances along the trip shape
        stop_distances = {
            stop_id: trip_shape.project(hlp.find_closest_point(trip_shape, stop_geometries[stop_id]))
            for stop_id in stop_times["stop_id"]
        }

        # Extract relevant numpy arrays for faster processing
        stop_ids = stop_times["stop_id"].to_numpy()
        arrival_times = stop_times["arrival_offset"].to_numpy()
        departure_times = stop_times["departure_offset"].to_numpy()

        # Compute travel distances and durations between stops
        stop_distances_arr = np.array([stop_distances[stop_id] for stop_id in stop_ids])
        travel_distances = np.diff(stop_distances_arr)  # Distance between consecutive stops
        travel_durations = np.maximum(0, arrival_times[1:] - departure_times[:-1])  # Ensure non-negative durations

        # --- Step 4: Build the travel sequence ---
        sequence = []
        cumulative_distance = 0
        cumulative_duration = 0

        # Add first stop (terminal)
        sequence.append({
            "status": "at_terminal",
            "distance": 0,
            "duration": departure_times[0] - arrival_times[0],
            "distance_from_start": cumulative_distance,
            "duration_from_start": cumulative_duration,
            "location": stop_geometries[stop_ids[0]]
        })
        cumulative_duration += departure_times[0] - arrival_times[0]

        # Process travel segments and intermediate stops
        for i in range(len(stop_ids) - 1):
            prev_geom = stop_geometries[stop_ids[i]]
            curr_geom = stop_geometries[stop_ids[i + 1]]

            start_distance = trip_shape.project(prev_geom)
            end_distance = trip_shape.project(curr_geom)

            # Handle cases where the shape loops around
            if start_distance > end_distance:
                sub_linestring1 = substring(trip_shape, start_distance, trip_shape.length)
                sub_linestring2 = substring(trip_shape, 0, end_distance)

                # Combine the two segments into a single LineString
                sub_linestring = LineString(list(sub_linestring1.coords) + list(sub_linestring2.coords))

                travel_distance_km = hlp.length_km(sub_linestring1) + hlp.length_km(sub_linestring2)
            elif start_distance != end_distance:
                sub_linestring = substring(trip_shape, start_distance, end_distance)
                travel_distance_km = hlp.length_km(sub_linestring)
            else:
                sub_linestring = None
                travel_distance_km = 0  # No movement

            # Add travelling event
            sequence.append({
                "status": "travelling",
                "distance": travel_distance_km,
                "duration": travel_durations[i],
                "distance_from_start": cumulative_distance,
                "duration_from_start": cumulative_duration,
                "location": sub_linestring  # LineString
            })
            cumulative_distance += travel_distance_km
            cumulative_duration += travel_durations[i]

            # Add stop event (either intermediate stop or terminal)
            sequence.append({
                "status": "at_terminal" if i + 1 == len(stop_ids) - 1 else "at_stop",
                "distance": 0,
                "duration": departure_times[i + 1] - arrival_times[i + 1],
                "distance_from_start": cumulative_distance,
                "duration_from_start": cumulative_duration,
                "location": prev_geom  # Point
            })
            cumulative_duration += departure_times[i + 1] - arrival_times[i + 1]

        # Store the computed sequence
        self._single_trip_sequence = sequence

    def compute_fleet_operation(self):
            """
            Computes the travel sequences for all vehicles operating throughout the day.

            This function determines how many vehicles are in operation based on the trip frequency data.
            It generates a sequence of travel events for each vehicle, accounting for staggered departures,
            repeated trips, and clipping of events that do not fully fit within the operational window.

            - Vehicles are staggered along their trip at the start of each frequency interval.
            - Trips are repeated as many times as possible within the interval.
            - If a trip starts within the interval but doesn't fully fit, it is still included with clipped times.

            The computed fleet operation sequence is stored in `self._fleet_operation` as a list of dictionaries,
            each containing:
                - vehicle_id: unique identifier for the vehicle
                - start_time: event start time (HH:MM:SS)
                - end_time: event end time (HH:MM:SS)
                - status: "at_terminal", "at_stop", or "travelling"
                - distance: distance traveled in km (0 for stops)
                - duration: duration of the event in seconds
                - distance_from_start: cumulative distance from trip start
                - duration_from_start: cumulative duration from trip start
            """
            # Compute the base trip sequence for a single vehicle.
            self.compute_single_trip_sequence()
            base_seq = self._single_trip_sequence

            # Compute total trip duration (sum of all event durations)
            trip_duration = sum(event['duration'] for event in base_seq)
            if trip_duration <= 0:
                raise ValueError("Trip duration must be positive.")

            # Determine the maximum number of vehicles that can operate simultaneously
            num_vehicles = self.max_vehicles_in_operation()

            # Convert base sequence into numpy arrays for efficient processing
            durations = np.array([event['duration'] for event in base_seq])
            cum_times = np.concatenate(([0], np.cumsum(durations)))  # Cumulative event start times
            event_status = np.array([event['status'] for event in base_seq])
            event_distance = np.array([
                event['distance'] if event['status'] == "travelling" else 0 for event in base_seq
            ])
            event_cum_distance = np.array([event['distance_from_start'] for event in base_seq])
            event_cum_duration = np.array([event['duration_from_start'] for event in base_seq])

            rows = []  # Output container for processed fleet events

            # Retrieve trip frequency intervals from GTFS data
            trip_freq = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]
            if trip_freq.empty:
                raise ValueError(f"No frequency data available for trip ID {self.trip_id}.")

            # Iterate over each frequency interval to schedule trips
            for _, freq in trip_freq.iterrows():
                headway = freq['headway_secs']  # Time between vehicle departures in seconds
                freq_start_sec = (freq['start_time'].hour * 3600 + freq['start_time'].minute * 60 + freq['start_time'].second)
                freq_end_sec = (freq['end_time'].hour * 3600 + freq['end_time'].minute * 60 + freq['end_time'].second)

                # Compute the number of vehicles required for the interval
                vehicles_in_operation = min(num_vehicles, max(1, round(trip_duration / headway)))
                active_vehicles = np.arange(vehicles_in_operation)  # Vehicle indices

                # Assign trips to vehicles
                for veh in active_vehicles:
                    vehicle_id = f"{self.trip_id}_{veh}"
                    initial_offset = veh * headway  # Stagger vehicle starts by headway interval
                    trip_start_time = freq_start_sec - initial_offset

                    # Repeat trips within the frequency interval
                    repetition_start = trip_start_time
                    while repetition_start < freq_end_sec:
                        event_abs_starts = repetition_start + cum_times[:-1]  # Absolute start times
                        event_abs_ends = repetition_start + cum_times[1:]  # Absolute end times

                        # Determine which events fall within the frequency interval
                        valid_mask = (event_abs_ends >= freq_start_sec) & (event_abs_starts <= freq_end_sec)
                        if not np.any(valid_mask):
                            break  # Stop if no events are valid

                        valid_starts = event_abs_starts[valid_mask]
                        valid_ends = event_abs_ends[valid_mask]
                        valid_durations = durations[valid_mask]
                        valid_status = event_status[valid_mask]
                        valid_distances = event_distance[valid_mask]
                        valid_cum_distances = event_cum_distance[valid_mask]
                        valid_cum_durations = event_cum_duration[valid_mask]

                        # Clip event start and end times to fit within the frequency interval
                        clip_starts = np.maximum(valid_starts, freq_start_sec)
                        clip_ends = np.minimum(valid_ends, freq_end_sec)
                        clipped_durations = clip_ends - clip_starts

                        # Compute proportional travel distances for clipped travel events
                        fractions = np.where(valid_durations > 0, clipped_durations / valid_durations, 0)
                        clipped_distances = valid_distances * fractions

                        # Compute updated cumulative distances and durations
                        clipped_cum_distances = valid_cum_distances + clipped_distances
                        clipped_cum_durations = valid_cum_durations + clipped_durations

                        # Convert timestamps to HH:MM:SS format
                        start_times_str = pd.to_datetime(clip_starts, unit="s").strftime("%H:%M:%S").tolist()
                        end_times_str = pd.to_datetime(clip_ends, unit="s").strftime("%H:%M:%S").tolist()

                        # Store each valid event in the results list
                        for i in range(len(clip_starts)):
                            rows.append({
                                "vehicle_id": vehicle_id,
                                "start_time": start_times_str[i],
                                "end_time": end_times_str[i],
                                "status": valid_status[i],
                                "distance": clipped_distances[i],
                                "duration": clipped_durations[i],
                                "distance_from_start": clipped_cum_distances[i],
                                "duration_from_start": clipped_cum_durations[i]
                            })

                        # Move to the next trip repetition
                        repetition_start += trip_duration

            # Store the computed fleet operation sequence
            self._fleet_operation = rows

    # Helper functions

    def trip_duration_sec(self) -> int:
        """Returns the duration of the trip in seconds."""
        trip_stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id]
        duration = (trip_stop_times['departure_time'].iloc[-1] - trip_stop_times['arrival_time'].iloc[0]).total_seconds()

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

    # Fleet trajectory

    def get_fleet_trajectory(self, time_step: int) -> pd.DataFrame:
        """
        Optimized simulation of fleet operation using the trip travel sequence.
        Now includes vehicle locations at each time step.
        
        Args:
            time_step (int): Time step in seconds.

        Returns:
            pd.DataFrame: DataFrame with vehicle status and location at each time step.
        """
        trip_duration = self.trip_duration_sec()
        num_vehicles = self.max_vehicles_in_operation()

        # Generate the time range for simulation
        time_range = pd.date_range("00:00:00", "23:59:59", freq=f"{time_step}s").time
        fleet_status = np.empty((num_vehicles, len(time_range)), dtype=object)
        fleet_location = np.empty((num_vehicles, len(time_range)), dtype=object)
        
        fleet_status.fill(None)
        fleet_location.fill(None)

        trip_frequencies = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]

        print(f"INFO \t Simulating fleet trajectory for trip \"{self.trip_id}\" with {len(trip_frequencies)} frequency intervals...")

        # Get base trip sequence
        # Check if fleet_travel_sequence exists and is not None
        if hasattr(self, "_single_trip_sequence") and self._single_trip_sequence is not None:
            base_seq = self._single_trip_sequence
        else:
            self.compute_fleet_operation()
            base_seq = self._single_trip_sequence
        
        event_durations = np.array([event["duration"] for event in base_seq])
        event_statuses = np.array([event["status"] for event in base_seq])
        event_locations = np.array([event["location"] for event in base_seq])
        event_cumulative_durations = np.concatenate(([0], np.cumsum(event_durations)))

        for _, row in trip_frequencies.iterrows():
            start_time, end_time = row["start_time"].time(), row["end_time"].time()
            headway = row["headway_secs"]
            vehicles_in_operation = max(1, round(trip_duration / headway))
            active_vehicles = np.arange(min(num_vehicles, vehicles_in_operation))

            # Efficiently determine time indices for the current frequency interval.
            time_range_sec = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_range])
            start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
            time_indices = np.where((time_range_sec >= start_sec) & (time_range_sec <= end_sec))[0]

            # Compute start offsets for vehicles.
            vehicle_start_offsets = np.arange(len(active_vehicles)) * headway
            vehicle_start_offsets[0] = 0  # Ensure the first vehicle starts at 0

            for veh, offset in zip(active_vehicles, vehicle_start_offsets):
                # Compute duration from start for each time index.
                durations = ((time_indices - time_indices[0]) * time_step - offset) % trip_duration

                # Find the corresponding status for each time step
                status_indices = np.maximum(0, np.searchsorted(event_cumulative_durations, durations, side="left") - 1)

                statuses = event_statuses[status_indices]
                locations = []

                for i, idx in enumerate(status_indices):
                    status = statuses[i]
                    if status in ["at_stop", "at_terminal"]:
                        locations.append(event_locations[idx])  # Store the Point directly
                    else:
                        # Compute the share of travel completed **between previous and next stop**
                        prev_stop_time = event_cumulative_durations[idx]
                        next_stop_time = event_cumulative_durations[idx + 1]
                        time_in_travel = durations[i] - prev_stop_time
                        total_travel_time = next_stop_time - prev_stop_time
                        travel_fraction = time_in_travel / total_travel_time if total_travel_time > 0 else 0

                        # Extract the correct position along the travel linestring
                        travel_linestring = event_locations[idx]
                        if isinstance(travel_linestring, LineString) and travel_linestring.length > 0:
                            travel_distance = travel_linestring.length * travel_fraction
                            travel_point = substring(travel_linestring, 0, travel_distance).interpolate(1.0, normalized=True)
                            locations.append(travel_point)
                        else:
                            locations.append(None)

                # Store results
                fleet_location[veh, time_indices] = locations

        print("\t Fleet trajectory simulation completed.")
        
        return pd.DataFrame(fleet_location, index=range(num_vehicles), columns=time_range)

    # Visualisation

    def map_fleet_trajectory(self, fleet_trajectory: pd.DataFrame) -> folium.Map:
        """
        Generates an interactive folium map with a time slider using the simulated fleet operation data.

        This function only works if the time step is exactly 2 minutes.

        Args:
            fleet_trajectory (pd.DataFrame): DataFrame containing vehicle trajectory data.

        Returns:
            folium.Map: A folium map with a time slider visualization of vehicle movements.
        """
        # Transform the fleet operation DataFrame into a long-format DataFrame
        data = []
        for vehicle_id in fleet_trajectory.index:
            for time_key, record in fleet_trajectory.loc[vehicle_id].items():
                if record is None or not isinstance(record, Point):
                    continue  # Skip invalid data

                # Extract lat/lon from the Point geometry
                longitude, latitude = record.x, record.y

                # Convert time_key to a datetime object
                if isinstance(time_key, str):
                    time_obj = datetime.strptime(time_key, "%H:%M:%S").time()
                else:
                    time_obj = time_key

                timestamp = datetime.combine(date.today(), time_obj)
                data.append({
                    "vehicle_id": vehicle_id,
                    "time": timestamp,
                    "latitude": latitude,
                    "longitude": longitude
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

        min_step = int(unique_times.diff().dropna().min().total_seconds() // 60)  # in minutes
        if min_step != 2:
            raise ValueError("Invalid time step. This function only works with a 2-minute time step.")

        # Create base map centered at the first valid position
        first_position = df.iloc[0]
        m = folium.Map(location=[first_position['latitude'], first_position['longitude']], zoom_start=12)

        # Add trip shape (polyline) to the map if available
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
            grouped_locations = group.groupby(["latitude", "longitude"])
            for (lat, lon), sub_group in grouped_locations:
                num_vehicles = len(sub_group)
                if num_vehicles > 1:
                    jitter_offsets = np.linspace(-0.0005, 0.0005, num_vehicles)  # Small lat/lon shifts
                    for (offset, row) in zip(jitter_offsets, sub_group.itertuples()):
                        vehicle_points.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon + offset, lat + offset]  # Apply small shift
                            },
                            "properties": {
                                "time": timestamp.isoformat(),
                                "popup": f"Vehicle {row.vehicle_id}\nTime: {timestamp.strftime('%H:%M:%S')}"
                            }
                        })
                else:
                    row = sub_group.iloc[0]
                    vehicle_points.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "time": timestamp.isoformat(),
                            "popup": f"Vehicle {row.vehicle_id}\nTime: {timestamp.strftime('%H:%M:%S')}"
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
        period="PT2M",
        duration="PT1M",
        add_last_point=False,
        auto_play=False,
        loop=True,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True).add_to(m)

        return m


