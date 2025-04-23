# coding: utf-8

import numpy as np
import pandas as pd
import random
from datetime import date, datetime, timedelta

from gtfs4ev.fleetsimulator import FleetSimulator

class ChargingSimulator:
    """A class to simulate the scenario-based charging of the EV fleet and the related implications."""

    def __init__(self, fleet_sim: FleetSimulator, energy_consumption_kWh_per_km: float, charging_efficiency: float = 0.9, charging_powers_kW: dict = None):
        """
        Initializes the Charging Simulator.

        Args:
            fleet_sim: An instance of a FleetSimulator class containing vehicle operations.
            vehicle_properties (dict): A dictionary of vehicle properties (e.g., energy consumption, battery capacity).
            charging_powers_kw (dict): A dictionary of vehicle_id: charging power in kW.
            charging_efficiency (float): Charging efficiency as a value between 0 and 1 (default is 0.0).
        """
        print("\n=========================================")
        print(f"INFO \t Creation of a ChargingSimulator object.")
        print("=========================================")

        self.fleet_sim = fleet_sim
        self.energy_consumption_kWh_per_km = energy_consumption_kWh_per_km
        self.charging_efficiency = charging_efficiency
        self.charging_powers_kW = charging_powers_kW or {}

        self._charging_schedule_pervehicle = None
        self._charging_schedule_perstop = None

        print("INFO \t Successful initialization of the ChargingSimulator. ")

    # --- fleet_sim ---
    @property
    def fleet_sim(self):
        return self._fleet_sim

    @fleet_sim.setter
    def fleet_sim(self, value):
        if value.fleet_operation is None:
            raise ValueError("fleet_sim.fleet_operation cannot be None. You must simulate the fleet opeation first.")
        self._fleet_sim = value

    @property
    def charging_schedule_pervehicle(self):
        """Gets the pd dataframe with charging schedule for every vehicle."""
        return self._charging_schedule_pervehicle

    @property
    def charging_schedule_perstop(self):
        """Gets the pd dataframe with charging schedule for every stop."""
        return self._charging_schedule_perstop

    def compute_charging_schedule(self, charging_strategy: str, **kwargs):
        """
        Reconstructs travel sequences for all vehicles.
        Then, for every vehicle, applies a charging scenario based on the provided charging function.

        Parameters:
        - charging_function (callable): A function that takes a travel sequence and additional parameters, and returns the charging sequence.
        - *charging_args (tuple): Additional arguments to be passed to the charging function.

        Returns:
        - DataFrame with vehicle_id, and a list of charging events as dictionaries.
        """
        print("INFO \t Computing the charging schedule for all vehicles... ")

        fleet_operation = self.fleet_sim.fleet_operation
        trip_travel_sequences = self.fleet_sim.trip_travel_sequences

        trip_base_sequences = {}
        for _, event in trip_travel_sequences.iterrows():
            trip_id = event["trip_id"]
            trip_base_sequences.setdefault(trip_id, []).append(event)

        vehicle_charging_sequences = []

        for _, vehicle_record in fleet_operation.iterrows():
            vehicle_id = vehicle_record["vehicle_id"]
            trip_id = vehicle_record.get("trip_id", None)
            vehicle_sequences = vehicle_record["travel_sequences"]
            base_seq = trip_base_sequences.get(trip_id, [])

            base_durations = [event["duration"] for event in base_seq]
            base_distances = [event.get("distance", 0.0) for event in base_seq]
            base_statuses = [event["status"] for event in base_seq]
            base_locations = [event.get("location", None) for event in base_seq]
            base_stop_ids = [event.get("stop_id", None) for event in base_seq]
            base_cumulative = np.concatenate(([0], np.cumsum(base_durations)))

            travel_sequence = []

            for seq in vehicle_sequences:
                start_time = datetime.strptime(seq["start_time"], "%H:%M:%S")
                end_time = datetime.strptime(seq["end_time"], "%H:%M:%S")
                if end_time < start_time:
                    end_time += timedelta(days=1)
                duration_sec = int((end_time - start_time).total_seconds())

                if not seq.get("operating", True):
                    travel_sequence.append({
                        "start_time": start_time.strftime("%H:%M:%S"),
                        "end_time": end_time.strftime("%H:%M:%S"),
                        "status": "not_operating",
                        "duration_h": duration_sec / 3600,
                        "distance_km": 0.0,
                        "stop_id": None
                    })
                    continue

                for i, event in enumerate(base_seq):
                    event_start_offset = base_cumulative[i]
                    event_end_offset = base_cumulative[i + 1]

                    if event_start_offset >= duration_sec:
                        break

                    effective_start = start_time + timedelta(seconds=event_start_offset)
                    effective_end = start_time + timedelta(seconds=min(event_end_offset, duration_sec))
                    event_duration = (effective_end - effective_start).total_seconds()
                    base_duration = base_durations[i]

                    base_distance = base_distances[i]
                    if base_duration > 0:
                        distance_km = base_distance * (event_duration / base_duration)
                    else:
                        distance_km = 0.0

                    travel_sequence.append({
                        "start_time": effective_start.strftime("%H:%M:%S"),
                        "end_time": effective_end.strftime("%H:%M:%S"),
                        "status": base_statuses[i],
                        "duration_h": event_duration / 3600,
                        "distance_km": distance_km,
                        "stop_id": base_stop_ids[i]
                    })

            # Calculate total charging need based on the travel distance, energy consumption, and charging efficiency
            charging_need_kWh = sum(event["distance_km"] * self.energy_consumption_kWh_per_km for event in travel_sequence)

            charging_events = []       # List to store accepted charging events
            remaining_need = charging_need_kWh
            added_intervals = []       # Tracks time intervals of already accepted charging events

            # Apply each charging strategy in order of priority until the energy need is fulfilled
            for strategy in charging_strategy:
                if remaining_need <= 0:
                    break  # Stop if no more energy is needed

                # Generate candidate charging events based on the current strategy
                events = self.get_charging_sequence(
                    travel_sequence, strategy, remaining_need, **kwargs
                )

                for event in events:
                    # Convert time strings to datetime objects for comparison
                    event_start_time = datetime.strptime(event["start_time"], "%H:%M:%S")
                    event_end_time = datetime.strptime(event["end_time"], "%H:%M:%S")

                    # Check if this event overlaps with any previously accepted event
                    overlaps = any(
                        not (event_end_time <= existing_start or event_start_time >= existing_end)
                        for (existing_start, existing_end) in added_intervals
                    )

                    if overlaps:
                        continue  # Skip overlapping events to preserve higher-priority decisions

                    # Accept this event: add it to the charging list and register its time interval
                    charging_events.append(event)
                    added_intervals.append((event_start_time, event_end_time))

                # Update the remaining energy need based on accepted charging events so far
                charged_energy = sum(e["energy_charged_kWh"] for e in charging_events)
                remaining_need = charging_need_kWh - charged_energy

            # Ensure charging events are returned in chronological order
            charging_events.sort(key=lambda e: e["start_time"])

            min_capacity = self.find_minimum_battery_capacity(travel_sequence, charging_events)

            vehicle_charging_sequences.append({
                "vehicle_id": vehicle_id,
                "charging_sequence": charging_events,
                "charging_need_kWh": charging_need_kWh,
                "remaining_need_kWh": remaining_need,
                "min_capacity_kWh": min_capacity
            })

        self._charging_schedule_pervehicle = pd.DataFrame(vehicle_charging_sequences)

        # Also calculate the charging schedule per stop
        self.aggregate_charging_by_stop()

    def get_charging_sequence(self, travel_sequence, charging_strategy, charging_need_kWh, charge_probability, depot_travel_time_min):
        """
        Simulates charging events based on the travel sequence and chosen strategy.
        Includes probabilistic arrival/departure offsets for depot time.
        """
        charging_events = []
        # Apply random delays to consider travel time to depot
        delay = timedelta(minutes=random.randint(depot_travel_time_min[0], depot_travel_time_min[1]))

        # --- Charging strategy: Terminal ---
        if charging_strategy == "terminal":
            remaining_need = charging_need_kWh 

            for event in travel_sequence:
                if remaining_need <= 0:
                    break  # charging need fulfilled

                if event["status"] == "at_terminal":
                    if random.random() < charge_probability:
                        power_rate = self.get_random_charging_power("stop")
                        duration_h = event["duration_h"]

                        # Max energy we can charge during this slot
                        max_possible_energy = duration_h * power_rate
                        energy_to_charge = min(max_possible_energy, remaining_need)
                        charging_duration_h = energy_to_charge / power_rate

                        actual_end = datetime.strptime(event["start_time"], "%H:%M:%S") + timedelta(hours=charging_duration_h)
                        
                        energy = charging_duration_h * power_rate

                        charging_events.append({
                            "start_time": event["start_time"],
                            "end_time": actual_end.strftime("%H:%M:%S"),
                            "location": "terminal",
                            "stop_id": event["stop_id"],
                            "power": power_rate,
                            "energy_charged_kWh": energy
                        })
                        remaining_need -= energy

        # --- Charging strategy: Stop ---
        if charging_strategy == "stop":
            remaining_need = charging_need_kWh 

            for event in travel_sequence:
                if remaining_need <= 0:
                    break  # charging need fulfilled

                if event["status"] == "at_stop":
                    if random.random() < charge_probability:
                        power_rate = self.get_random_charging_power("stop")
                        duration_h = event["duration_h"]

                        # Max energy we can charge during this slot
                        max_possible_energy = duration_h * power_rate
                        energy_to_charge = min(max_possible_energy, remaining_need)
                        charging_duration_h = energy_to_charge / power_rate

                        actual_end = datetime.strptime(event["start_time"], "%H:%M:%S") + timedelta(hours=charging_duration_h)
                        
                        energy = charging_duration_h * power_rate

                        charging_events.append({
                            "start_time": event["start_time"],
                            "end_time": actual_end.strftime("%H:%M:%S"),
                            "location": "stop",
                            "stop_id": event["stop_id"],
                            "power": power_rate,
                            "energy_charged_kWh": energy
                        })
                        remaining_need -= energy

        # --- Charging strategy: Depot day ---
        if charging_strategy == "depot_day":
            remaining_need = charging_need_kWh            

            for idx, event in enumerate(travel_sequence):             
                
                # Skip first and last events
                if idx == 0 or idx == len(travel_sequence) - 1:
                    continue

                if event["status"] == "not_operating" and remaining_need > 0:
                    power_rate = self.get_random_charging_power("depot")

                    start_dt = datetime.strptime(event["start_time"], "%H:%M:%S")
                    end_dt = datetime.strptime(event["end_time"], "%H:%M:%S")                    

                    adjusted_start = start_dt + delay
                    adjusted_end = end_dt - delay

                    if adjusted_end < adjusted_start:
                        continue

                    # Available duration in hours
                    available_duration_h = (adjusted_end - adjusted_start).total_seconds() / 3600

                    if available_duration_h > 0:
                        # Max energy we can charge during this slot
                        max_possible_energy = available_duration_h * power_rate
                        energy_to_charge = min(max_possible_energy, remaining_need)
                        charging_duration_h = energy_to_charge / power_rate

                        actual_end = adjusted_start + timedelta(hours=charging_duration_h)

                        charging_events.append({
                            "start_time": adjusted_start.strftime("%H:%M:%S"),
                            "end_time": actual_end.strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_to_charge
                        })

                        remaining_need -= energy_to_charge
                        if remaining_need <= 0:
                            break  # charging need fulfilled

        # --- Charging strategy: Depot night ---
        if charging_strategy == "depot_night":
            remaining_need = charging_need_kWh
                
            power_rate = self.get_random_charging_power("depot") # One power rate for both last and first sequences

            if remaining_need > 0:
                # --- End of Day ---
                last_event = travel_sequence[-1]
                start_dt = datetime.strptime(last_event["start_time"], "%H:%M:%S")
                end_dt = datetime.strptime(last_event["end_time"], "%H:%M:%S")

                adjusted_start = start_dt + delay
                duration_needed = timedelta(hours=remaining_need / power_rate)

                if adjusted_start + duration_needed <= end_dt:
                    charging_events.append({
                        "start_time": adjusted_start.strftime("%H:%M:%S"),
                        "end_time": (adjusted_start + duration_needed).strftime("%H:%M:%S"),
                        "location": "depot",
                        "power": power_rate,
                        "energy_charged_kWh": remaining_need
                    })
                    remaining_need = 0
                else:
                    available_h = (end_dt - adjusted_start).total_seconds() / 3600
                    if available_h > 0:
                        energy_end_of_day = available_h * power_rate
                        charging_events.append({
                            "start_time": adjusted_start.strftime("%H:%M:%S"),
                            "end_time": end_dt.strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_end_of_day
                        })
                        remaining_need -= energy_end_of_day

                # --- Beginning of Day ---
                if remaining_need > 0:
                    first_event = travel_sequence[0]
                    start_dt = datetime.strptime(first_event["start_time"], "%H:%M:%S")
                    end_dt = datetime.strptime(first_event["end_time"], "%H:%M:%S")

                    adjusted_end = end_dt - delay

                    available_h = (adjusted_end - start_dt).total_seconds() / 3600
                    if available_h > 0:
                        energy_morning = min(available_h * power_rate, remaining_need)
                        actual_duration = timedelta(hours=energy_morning / power_rate)

                        charging_events.append({
                            "start_time": start_dt.strftime("%H:%M:%S"),
                            "end_time": (start_dt + actual_duration).strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_morning
                        })

                        remaining_need -= energy_morning

        charging_events.sort(key=lambda e: e["start_time"])                

        return charging_events

    # Charging sequence with stop perspective

    def aggregate_charging_by_stop(self):
        """
        Aggregates charging sessions into non-overlapping intervals per stop_id,
        calculating power, energy, and number of vehicles charging.

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

        vehicle_df = self.charging_schedule_pervehicle

        stop_sessions = {}
        stop_locations = {}

        for _, row in vehicle_df.iterrows():
            for session in row['charging_sequence']:
                stop_id = session.get('stop_id', 'depot')
                location = session['location']

                if stop_id not in stop_sessions:
                    stop_sessions[stop_id] = []
                    stop_locations[stop_id] = location

                start = datetime.strptime(session['start_time'], '%H:%M:%S')
                end = datetime.strptime(session['end_time'], '%H:%M:%S')
                stop_sessions[stop_id].append({
                    'start': start,
                    'end': end,
                    'power': session['power']
                })

        result = []
        for stop_id, sessions in stop_sessions.items():
            time_points = set()
            for s in sessions:
                time_points.add(s['start'])
                time_points.add(s['end'])

            sorted_times = sorted(time_points)
            charging_sequence = []

            for i in range(len(sorted_times) - 1):
                t_start = sorted_times[i]
                t_end = sorted_times[i + 1]

                active_sessions = [
                    s for s in sessions if s['start'] <= t_start < s['end']
                ]

                vehicle_count = len(active_sessions)
                total_power = sum(s['power'] for s in active_sessions)
                duration_h = (t_end - t_start).total_seconds() / 3600
                energy_kWh = round(total_power * duration_h, 6)

                if total_power > 0:
                    charging_sequence.append({
                        'start_time': t_start.strftime('%H:%M:%S'),
                        'end_time': t_end.strftime('%H:%M:%S'),
                        'power': total_power,
                        'energy_kWh': energy_kWh,
                        'vehicle_count': vehicle_count
                    })

            # New summary metrics
            total_energy_kWh = round(sum(e['energy_kWh'] for e in charging_sequence), 6)
            peak_power_kW = max((e['power'] for e in charging_sequence), default=0)
            max_vehicles = max((e['vehicle_count'] for e in charging_sequence), default=0)

            result.append({
                'stop_id': stop_id,
                'location': stop_locations[stop_id],
                'charging_sequence': charging_sequence,
                'total_energy_kWh': total_energy_kWh,
                'peak_power_kW': peak_power_kW,
                'max_vehicles': max_vehicles
            })

        self._charging_schedule_perstop = pd.DataFrame(result)

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

    # Helpers

    def get_random_charging_power(self, location_type: str) -> float:
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

    def find_minimum_battery_capacity(self, travel_sequence, charging_events):
        def to_time(t): return datetime.strptime(t, "%H:%M:%S")

        # Merge and sort all events by time
        events = []

        # Energy consumption 
        energy_consumption_kWh_per_km = self.energy_consumption_kWh_per_km

        for event in travel_sequence:
            if event["status"] == "travelling":
                energy = -event["distance_km"] * energy_consumption_kWh_per_km
                events.append((to_time(event["start_time"]), energy))

        for event in charging_events:
            energy = event["energy_charged_kWh"]
            events.append((to_time(event["start_time"]), energy))

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