# coding: utf-8

import numpy as np
import pandas as pd
import random
from datetime import date, datetime, timedelta

from gtfs4ev.fleetsimulator import FleetSimulator

class ChargingSimulator:
    """
    A class simulating the charging of electric vehicles based on the mobility simulation and charging strategy."""

    def __init__(self, fleet_sim: FleetSimulator, energy_consumption_kWh_per_km: float, charging_powers_kW: dict = None):
        """
        Initializes the Charging Simulator.

        Args:
            fleet_sim: An instance of a FleetSimulator class containing vehicle operations.
            vehicle_properties (dict): A dictionary of vehicle properties (e.g., energy consumption, battery capacity).
            charging_powers_kw (dict): A dictionary of vehicle_id: charging power in kW.
        """
        print("=========================================")
        print(f"INFO \t Creation of a ChargingSimulator object.")
        print("=========================================")

        self.fleet_sim = fleet_sim
        self.energy_consumption_kWh_per_km = energy_consumption_kWh_per_km
        self.charging_powers_kW = charging_powers_kW or {}

        self._charging_schedule_pervehicle = None
        self._charging_schedule_perstop = None

        print("INFO \t Successful initialization of the ChargingSimulator. ")

    # Properties and Setters

    @property
    def fleet_sim(self):
        return self._fleet_sim

    @fleet_sim.setter
    def fleet_sim(self, value):
        if value.fleet_operation is None:
            raise ValueError("fleet_sim.fleet_operation cannot be None. You must simulate the fleet opeation first.")
        self._fleet_sim = value

    @property
    def energy_consumption_kWh_per_km(self):
        return self._energy_consumption_kWh_per_km

    @energy_consumption_kWh_per_km.setter
    def energy_consumption_kWh_per_km(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("energy_consumption_kWh_per_km must be a positive number.")
        self._energy_consumption_kWh_per_km = value

    @property
    def charging_powers_kW(self):
        return self._charging_powers_kW

    @charging_powers_kW.setter
    def charging_powers_kW(self, value):
        if not isinstance(value, dict):
            raise ValueError("charging_powers_kW must be a dictionary with location types as keys.")
        
        for location, power_list in value.items():
            if not isinstance(location, str):
                raise ValueError("Keys of charging_powers_kW must be strings representing location types.")
            if not isinstance(power_list, list):
                raise ValueError(f"Value for '{location}' must be a list of [power_kW, share] pairs.")
            for entry in power_list:
                if not (isinstance(entry, list) and len(entry) == 2 and 
                        isinstance(entry[0], (int, float)) and entry[0] > 0 and
                        isinstance(entry[1], (int, float)) and 0 <= entry[1] <= 1):
                    raise ValueError(f"Each entry in '{location}' must be [positive power_kW, share (0 to 1)].")

        self._charging_powers_kW = value

    @property
    def charging_schedule_pervehicle(self):
        """Gets the pd dataframe with charging schedule for every vehicle."""
        return self._charging_schedule_pervehicle

    @property
    def charging_schedule_perstop(self):
        """Gets the pd dataframe with charging schedule for every stop."""
        return self._charging_schedule_perstop

    # Charging schedule

    def compute_charging_schedule(self, charging_strategies: list[str], **kwargs):
        """
        Computes the vehicle-level and stop-level charging schedules using a list of charging strategies.

        For each vehicle:
        1. Reconstructs the detailed travel sequence based on operational data and predefined trip structure.
        2. Estimates the total energy need based on distance traveled and energy consumption rate.
        3. Applies each charging strategy (in priority order) to generate charging events.
        4. Accepts new charging events until the energy need is met.
        5. Records the charging events, energy requirement, remaining unmet energy, and minimum required battery capacity.

        Charging strategies are provided as a list of string identifiers, each corresponding to a known strategy
        supported by the `generate_charging_events_for_strategy` method.

        The method also computes stop-level charging statistics by aggregating all vehicle charging events.

        Parameters:
        - charging_strategy (list[str]): Ordered list of strategy names to apply for charging (e.g., ["depot", "terminal", "opportunity"]).
        - **kwargs: Additional keyword arguments passed to each strategy handler.

        Returns:
        - Updates the internal attributes:
            - self._charging_schedule_pervehicle: DataFrame with per-vehicle charging schedules.
            - self.aggregate_charging_by_stop(): Updates internal stop-level charging summary.
        """
        print("INFO \t Computing the charging schedule for all vehicles...")

        trip_base_sequences = self._group_trip_events_by_id()
        vehicle_charging_sequences = []

        for _, vehicle_record in self.fleet_sim.fleet_operation.iterrows():
            travel_sequence = self._construct_travel_sequence(vehicle_record, trip_base_sequences)            
            charging_result = self._apply_charging_strategies(travel_sequence, charging_strategies, **kwargs)            
            result = {"vehicle_id": vehicle_record["vehicle_id"], **charging_result} # Put vehicle_id first by manually ordering the dict
            vehicle_charging_sequences.append(result)

        self._charging_schedule_pervehicle = pd.DataFrame(vehicle_charging_sequences)
        self._charging_schedule_perstop = self._aggregate_charging_by_stop()

    def _group_trip_events_by_id(self):
        """Groups travel events by trip_id into a dictionary."""
        grouped = {}
        for _, event in self.fleet_sim.trip_travel_sequences.iterrows():
            trip_id = event["trip_id"]
            grouped.setdefault(trip_id, []).append(event)
        return grouped

    def _construct_travel_sequence(self, vehicle_record, trip_base_sequences):
        """Reconstructs a detailed travel sequence from the base and actual sequences."""
        trip_id = vehicle_record.get("trip_id", None)
        vehicle_sequences = vehicle_record["travel_sequences"]
        base_seq = trip_base_sequences.get(trip_id, [])

        base_durations = [event["duration"] for event in base_seq]
        base_distances = [event.get("distance", 0.0) for event in base_seq]
        base_statuses = [event["status"] for event in base_seq]
        base_stop_ids = [event.get("stop_id", None) for event in base_seq]
        base_cumulative = np.concatenate(([0], np.cumsum(base_durations)))

        travel_sequence = []

        for seq in vehicle_sequences:
            travel_sequence.extend(
                self._generate_segment_sequence(seq, base_seq, base_durations, base_distances, base_statuses, base_stop_ids, base_cumulative)
            )

        return travel_sequence

    def _generate_segment_sequence(self, seq, base_seq, base_durations, base_distances, base_statuses, base_stop_ids, base_cumulative):
        """Breaks a single time interval into base-sequence-aligned segments."""
        start_time = datetime.strptime(seq["start_time"], "%H:%M:%S")
        end_time = datetime.strptime(seq["end_time"], "%H:%M:%S")
        if end_time < start_time:
            end_time += timedelta(days=1)
        duration_sec = int((end_time - start_time).total_seconds())

        if not seq.get("operating", True):
            return [{
                "start_time": start_time.strftime("%H:%M:%S"),
                "end_time": end_time.strftime("%H:%M:%S"),
                "status": "not_operating",
                "duration_h": duration_sec / 3600,
                "distance_km": 0.0,
                "stop_id": None
            }]

        segments = []
        for i, _ in enumerate(base_seq):
            if base_cumulative[i] >= duration_sec:
                break

            start_offset = base_cumulative[i]
            end_offset = base_cumulative[i + 1]
            effective_start = start_time + timedelta(seconds=start_offset)
            effective_end = start_time + timedelta(seconds=min(end_offset, duration_sec))
            event_duration = (effective_end - effective_start).total_seconds()

            base_duration = base_durations[i]
            base_distance = base_distances[i]
            distance_km = base_distance * (event_duration / base_duration) if base_duration > 0 else 0.0

            segments.append({
                "start_time": effective_start.strftime("%H:%M:%S"),
                "end_time": effective_end.strftime("%H:%M:%S"),
                "status": base_statuses[i],
                "duration_h": event_duration / 3600,
                "distance_km": distance_km,
                "stop_id": base_stop_ids[i]
            })

        return segments

    def _apply_charging_strategies(self, travel_sequence, charging_strategy, **kwargs):
        """Applies multiple charging strategies and returns charging data."""
        total_need = sum(event["distance_km"] * self.energy_consumption_kWh_per_km for event in travel_sequence)
        remaining = total_need
        charging_events = []
        added_intervals = []

        for strategy in charging_strategy:
            if remaining <= 0:
                break
            events = self._generate_charging_events_for_strategy(travel_sequence, strategy, remaining, **kwargs)
            for event in events:
                start = datetime.strptime(event["start_time"], "%H:%M:%S")
                end = datetime.strptime(event["end_time"], "%H:%M:%S")
                if any(not (end <= s or start >= e) for s, e in added_intervals):
                    continue
                charging_events.append(event)
                added_intervals.append((start, end))

            charged_energy = sum(e["energy_charged_kWh"] for e in charging_events)
            remaining = total_need - charged_energy

        charging_events.sort(key=lambda e: e["start_time"])
        min_capacity = self._find_minimum_battery_capacity(travel_sequence, charging_events)

        return {
            "charging_sequence": charging_events,
            "charging_need_kWh": total_need,
            "remaining_need_kWh": remaining,
            "min_capacity_kWh": min_capacity
        }

    def _generate_charging_events_for_strategy(self, 
        travel_sequence, 
        charging_strategy, 
        charging_need_kWh, 
        charge_probability, 
        depot_travel_time_min):
        """
        Simulates charging events based on a given strategy and a vehicle’s travel sequence.

        The function iterates through the vehicle's travel timeline and simulates potential charging sessions
        depending on the charging strategy selected. 

        Supported strategies:
        - "terminal": Attempts charging during 'at_terminal' statuses with a certain probability.
        - "stop": Attempts charging during 'at_stop' statuses with a certain probability.
        - "depot_day": Tries charging during daytime idle (non-operating) periods, excluding first/last events.
        - "depot_night": Charges either at the end or beginning of the day (or both), based on depot availability.

        Returns:
            List[Dict] of charging events with timing, energy amount, and location metadata.
        """
        charging_events = []
        remaining_need = charging_need_kWh  # How much energy still needs to be charged
        delay = timedelta(minutes=random.randint(*depot_travel_time_min))  # Random travel time offset

        # Helper to decide probabilistic charging
        def can_charge(prob):
            return random.random() < prob

        # Helper to compute a single charging event
        def compute_charging_event(start, duration_h, location, stop_id=None):
            power = self._get_random_charging_power(location)
            energy = min(duration_h * power, remaining_need)
            actual_duration_h = energy / power
            actual_end = datetime.strptime(start, "%H:%M:%S") + timedelta(hours=actual_duration_h)

            return {
                "start_time": start,
                "end_time": actual_end.strftime("%H:%M:%S"),
                "location": location,
                "stop_id": stop_id,
                "power": power,
                "energy_charged_kWh": energy
            }, energy

        # --- Charging Strategy: Terminal ---
        # Charge opportunistically when a vehicle is parked at the terminal (usually end/start of line).
        if charging_strategy == "terminal":
            for event in travel_sequence:
                if remaining_need <= 0:
                    break  # Stop if enough energy has been charged
                if event["status"] == "at_terminal" and can_charge(charge_probability):
                    charge_event, charged = compute_charging_event(
                        event["start_time"],
                        event["duration_h"],
                        "terminal",
                        event["stop_id"]
                    )
                    charging_events.append(charge_event)
                    remaining_need -= charged

        # --- Charging Strategy: Stop ---
        # Similar to terminal strategy, but happens at intermediate stops.
        elif charging_strategy == "stop":
            for event in travel_sequence:
                if remaining_need <= 0:
                    break
                if event["status"] == "at_stop" and can_charge(charge_probability):
                    charge_event, charged = compute_charging_event(
                        event["start_time"],
                        event["duration_h"],
                        "stop",
                        event["stop_id"]
                    )
                    charging_events.append(charge_event)
                    remaining_need -= charged

        # --- Charging Strategy: Depot Day ---
        # Use idle "not_operating" periods during the day (typically midday breaks) for depot charging.
        # Skips the first and last events to avoid overlap with depot_night strategy.
        elif charging_strategy == "depot_day":
            for idx, event in enumerate(travel_sequence):
                if idx in (0, len(travel_sequence) - 1):
                    continue  # Skip first/last event
                if event["status"] == "not_operating" and remaining_need > 0:
                    start_dt = datetime.strptime(event["start_time"], "%H:%M:%S") + delay
                    end_dt = datetime.strptime(event["end_time"], "%H:%M:%S") - delay

                    if end_dt <= start_dt:
                        continue  # Ignore negative or zero-length windows

                    duration_h = (end_dt - start_dt).total_seconds() / 3600
                    charge_event, charged = compute_charging_event(
                        start_dt.strftime("%H:%M:%S"),
                        duration_h,
                        "depot"
                    )
                    charging_events.append(charge_event)
                    remaining_need -= charged
                    if remaining_need <= 0:
                        break

        # --- Charging Strategy: Depot Night ---
        # Charge either at the end of the day or beginning of the next day (or both).
        elif charging_strategy == "depot_night":
            power = self._get_random_charging_power("depot")

            # End-of-day charging: last event
            if remaining_need > 0:
                last_event = travel_sequence[-1]
                start_dt = datetime.strptime(last_event["start_time"], "%H:%M:%S") + delay
                end_dt = datetime.strptime(last_event["end_time"], "%H:%M:%S")

                duration_h = (end_dt - start_dt).total_seconds() / 3600
                if duration_h > 0:
                    energy = min(duration_h * power, remaining_need)
                    duration_needed = timedelta(hours=energy / power)

                    charging_events.append({
                        "start_time": start_dt.strftime("%H:%M:%S"),
                        "end_time": (start_dt + duration_needed).strftime("%H:%M:%S"),
                        "location": "depot",
                        "power": power,
                        "energy_charged_kWh": energy
                    })
                    remaining_need -= energy

            # Beginning-of-day charging: first event
            if remaining_need > 0:
                first_event = travel_sequence[0]
                start_dt = datetime.strptime(first_event["start_time"], "%H:%M:%S")
                end_dt = datetime.strptime(first_event["end_time"], "%H:%M:%S") - delay

                duration_h = (end_dt - start_dt).total_seconds() / 3600
                if duration_h > 0:
                    energy = min(duration_h * power, remaining_need)
                    duration_needed = timedelta(hours=energy / power)

                    charging_events.append({
                        "start_time": start_dt.strftime("%H:%M:%S"),
                        "end_time": (start_dt + duration_needed).strftime("%H:%M:%S"),
                        "location": "depot",
                        "power": power,
                        "energy_charged_kWh": energy
                    })
                    remaining_need -= energy

        # Ensure chronological order of events
        charging_events.sort(key=lambda e: e["start_time"])

        return charging_events

    def _aggregate_charging_by_stop(self):
        """
        Aggregates individual vehicle charging sessions into summarized charging intervals per stop.
        Each interval captures the total power, energy used, and number of vehicles charging concurrently.

        Returns
        -------
        pd.DataFrame
            One row per stop_id with:
            - stop_id: str
            - location: str
            - charging_sequence: List[Dict] with:
                - 'start_time', 'end_time', 'power', 'energy_kWh', 'vehicle_count'
            - total_energy_kWh: float
            - peak_power_kW: float
            - max_vehicles: int
        """
        print("INFO \t Computing the charging schedule per stop location... ")

        # Retrieve the per-vehicle charging schedule DataFrame
        vehicle_df = self.charging_schedule_pervehicle

        stop_sessions = {}    # Dictionary to store charging sessions by stop_id
        stop_locations = {}   # Dictionary to map stop_id to physical location

        # Iterate over each vehicle's charging schedule
        for _, row in vehicle_df.iterrows():
            for session in row['charging_sequence']:
                stop_id = session.get('stop_id', 'depot')  # Default to 'depot' if stop_id is missing
                location = session['location']

                # Initialize data structures if encountering stop_id for the first time
                if stop_id not in stop_sessions:
                    stop_sessions[stop_id] = []
                    stop_locations[stop_id] = location

                # Convert session start and end times to datetime objects
                start = datetime.strptime(session['start_time'], '%H:%M:%S')
                end = datetime.strptime(session['end_time'], '%H:%M:%S')

                # Store session details for aggregation
                stop_sessions[stop_id].append({
                    'start': start,
                    'end': end,
                    'power': session['power']
                })

        result = []

        # Process each stop's sessions to create non-overlapping aggregated intervals
        for stop_id, sessions in stop_sessions.items():
            time_points = set()

            # Gather all unique time points (session starts and ends)
            for s in sessions:
                time_points.add(s['start'])
                time_points.add(s['end'])

            sorted_times = sorted(time_points)  # Sort time points chronologically
            charging_sequence = []

            # Create time intervals from sorted time points
            for i in range(len(sorted_times) - 1):
                t_start = sorted_times[i]
                t_end = sorted_times[i + 1]

                # Identify sessions active during the current interval
                active_sessions = [
                    s for s in sessions if s['start'] <= t_start < s['end']
                ]

                vehicle_count = len(active_sessions)
                total_power = sum(s['power'] for s in active_sessions)
                duration_h = (t_end - t_start).total_seconds() / 3600  # Convert duration to hours
                energy_kWh = round(total_power * duration_h, 6)

                # Add to the charging sequence if there is power being drawn
                if total_power > 0:
                    charging_sequence.append({
                        'start_time': t_start.strftime('%H:%M:%S'),
                        'end_time': t_end.strftime('%H:%M:%S'),
                        'power': total_power,
                        'energy_kWh': energy_kWh,
                        'vehicle_count': vehicle_count
                    })

            # Calculate summary statistics for the stop
            total_energy_kWh = round(sum(e['energy_kWh'] for e in charging_sequence), 6)
            peak_power_kW = max((e['power'] for e in charging_sequence), default=0)
            max_vehicles = max((e['vehicle_count'] for e in charging_sequence), default=0)

            # Append results for this stop
            result.append({
                'stop_id': stop_id,
                'location': stop_locations[stop_id],
                'charging_sequence': charging_sequence,
                'total_energy_kWh': total_energy_kWh,
                'peak_power_kW': peak_power_kW,
                'max_vehicles': max_vehicles
            })

        # Convert results to a DataFrame and return
        return pd.DataFrame(result)

    def _get_random_charging_power(self, location_type: str):
        """
        Randomly selects a charging power based on the location type and its distribution.
            Args:
            location_type (str): The type of location ("depot", "terminal", etc.)

        Returns:
            float: Selected charging power in kW.
        """
        if location_type not in self.charging_powers_kW:
            raise ValueError(f"Unknown location type '{location_type}'")

        options = self.charging_powers_kW[location_type]
        powers = [item[0] for item in options]
        probabilities = [item[1] for item in options]

        return random.choices(powers, weights=probabilities, k=1)[0]

    def _find_minimum_battery_capacity(self, travel_sequence, charging_events):
        # Merge and sort all events by time
        events = []

        # Energy consumption 
        energy_consumption_kWh_per_km = self.energy_consumption_kWh_per_km

        for event in travel_sequence:
            if event["status"] == "travelling":
                energy = -event["distance_km"] * energy_consumption_kWh_per_km
                events.append( (datetime.strptime(event["start_time"], "%H:%M:%S"), energy) )

        for event in charging_events:
            energy = event["energy_charged_kWh"]
            events.append( (datetime.strptime(event["start_time"], "%H:%M:%S"), energy) )

        # Sort all events chronologically
        events.sort(key=lambda x: x[0])

        # Simulate battery usage
        battery_level = 0.0
        min_battery_level = 0.0
        max_battery_level = 0.0

        for time, energy in events:
            battery_level += energy
            min_battery_level = min(min_battery_level, battery_level)
            max_battery_level = max(max_battery_level, battery_level)

        # Battery profile is shifted so the minimum is 0
        required_capacity = max_battery_level - min_battery_level

        return required_capacity

    # Charging load curve

    def compute_charging_load_curve(self, time_step_s):
        """
        Compute aggregated charging load curves by location from vehicle charging sequences.

        Parameters:
        - time_step_s: int, the resolution of the output time series in seconds

        Returns:
        - pd.DataFrame: charging load with columns:
            - 'time_h': time elapsed from midnight in hours
            - One column per location: aggregated power [kW] at each time step
          Index is datetime.time (HH:MM:SS)
        """
        print("INFO \t Computing the charging load curve... ")

        df = self.charging_schedule_pervehicle

        # Full day index at desired resolution
        full_day = pd.date_range(
            start="00:00:00", 
            end="23:59:59", 
            freq=f"{time_step_s}s"
        )

        # Will store location-wise power curves
        location_curves = {}

        for _, row in df.iterrows():
            for session in row['charging_sequence']:
                start_dt = datetime.strptime(session['start_time'], '%H:%M:%S')
                end_dt = datetime.strptime(session['end_time'], '%H:%M:%S')
                power = session['power']
                location = session['location']

                if location not in location_curves:
                    location_curves[location] = pd.Series(0.0, index=full_day)

                session_range = pd.date_range(
                    start=start_dt.time().strftime('%H:%M:%S'),
                    end=end_dt.time().strftime('%H:%M:%S'),
                    freq=f'{time_step_s}s'
                )

                for t in session_range:
                    if t in location_curves[location].index:
                        location_curves[location][t] += power

        # Build full DataFrame
        load_df = pd.DataFrame(location_curves, index=full_day)

        # Add time in hours as first column
        time_h = load_df.index.hour + load_df.index.minute / 60 + load_df.index.second / 3600
        load_df.insert(0, 'time_h', time_h)

        # Set index to time only (HH:MM:SS)
        load_df.index = load_df.index.time

        return load_df