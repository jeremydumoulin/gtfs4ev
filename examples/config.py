# ------------------------------------------
# CONFIGURATION FILE: GTFS4EV MODEL
# ------------------------------------------

# ========================================== #
# ============= BASIC PARAMETERS =========== #
# ========================================== #

# --- General ---
output_folder = "output"  # Output folder path (stores the output files)
scenario_name = "Nairobi"  # Scenario label (used in output filenames)

# --- GTFS Data ---
gtfs_datafolder = "input/GTFS_Nairobi"

# --- Fleet Operation simulation ---
trips_to_simulate = ["1011F110", "1107D110", "10114111"] # If None, simulates all trips. You can also simulate only a specific number of trips by putting a 
# list of trips ids to simulate trips_to_simulate = ["id1", "id2", etc.]

# --- Charging Scenario ---
energy_consumption_kWh_per_km = 0.39  # Energy used per km (in kWh)
charging_powers_kW = {
    "depot": [[11, 1.0], [22, 0.0]],
    "terminal": [[100, 1.0]],
    "stop": [[200, 1.0]]
}

# Charging strategy sequence (in order of application)
# The model start by applying the first strategy. If the charging needs of a 
# vehicle are not met, the next one is tried. And so on, until the vehicle is 
# sufficiently charged or all strategies are exhausted.

# Available strategies:
# - "terminal_random"       → Random charging at terminal with given probability
# - "stop_random"           → Random charging at regular stops with given probability
# - "depot_night"           → Recharge at depot during off-hours at the end of the day
# - "depot_day"             → Recharge at depot during the day when not in operation

charging_strategy_sequence = ["terminal_random", "stop_random", "depot_night"]

# Strategy-specific parameters
charge_probability_terminal = 0.1
charge_probability_stop = 0.1
depot_travel_time_min = [15, 30]  # Min/max travel time to depot in minutes

# --- PV Production and EV-PV Complementarity ---
latitude = 0.17094549
longitude = 37.9039685
year = 2020 # Year to simulate

installation_type = "groundmounted_fixed"  
# Options: 'rooftop', 'groundmounted_fixed', 'groundmounted_singleaxis_horizontal', 
#          'groundmounted_singleaxis_vertical', 'groundmounted_dualaxis'

pv_capacity_MW = 10 # Installed nominal PV capacity
start_date = "01-01" # Start date of the simulation (Format: MM-DD)
end_date = "01-07" # End date of the simulation (Format: MM-DD)

# ========================================== #
# =========== ADVANCED PARAMETERS ========== #
# ========================================== #

# --- GTFS Data Preprocessing ---
clean_gtfs_data = True
trim_tripshapes_to_terminal_locations = True
filter_out_services = []
filter_out_agencies = []
add_idle_time_terminals_s = [0, 0]
add_idle_time_stops_s = [0, 0]

# --- GTFS Outputs ---
export_cleaned_gtfs = False
generate_network_map = False
generate_map_specific_trips = ["1011F110"]

# --- Fleet operation simulation ---
use_multiprocessing = False
generate_map_fleet_movement = False # Warning: will only work if the number of trips to simulate is smaller than 5

# --- Charging Scenario ---
security_driving_distance_km = 0  # Extra buffer distance per trip (in km)
load_curve_timestep_s = 60

# --- PV Production ---
module_efficiency = 0.22
temperature_coefficient = -0.004
system_losses = 0.14