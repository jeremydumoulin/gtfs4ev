
# GTFS4EV
**GTFS4EV (GTFS for Electric Vehicles) is an open-source Python tool that supports planning the electrification of public transport systems. It simulates electric public transport operations and charging scenarios using widely available GTFS data. 
The model allows users to explore different charging strategies and assess their impacts on the charging needs in time and space, assessing also whether a specific charging strategy can meet the demand and what battery capacity is needed. It also evaluates how local solar PV energy could be used to fulfill the charging needs.**

Authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

Lead institution = EPFL PV-LAB, Switzerland

Contact = jeremy.dumoulin@epfl.ch 

Langage = python 3 

> :bulb: This `README.md` provides a quick start guide for basic usage of the model. Comprehensive documentation for detailed and advanced usage will soon be available on a [Read the Docs](https://readthedocs.org/) page. 

## Table of Contents

1. [Overview of the Model](#overview-of-the-model)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [Contributing](#contributing)
7. [Scientific Publications](#scientific-publications)
8. [Acknowledgment](#acknowledgment)
9. [License](#license)

## Overview of the model

1. **GTFS Data Preprocessing**. GTFS data is loaded, checked, cleaned, and optionally filtered or enriched (e.g., by adding idle times at terminals).  
*Main outputs*: Cleaned GTFS data, transport network map.

2. **Fleet Operation Simulation**.Vehicle movements are simulated based on the GTFS feed, either for the entire network or selected trips.  
*Main output*: Operational data for each vehicle.

3. **Charging Scenario Simulation**. Charging needs are computed using customizable strategies and charging infrastructure setups.  
*Main outputs*: Charging schedules per vehicle, estimated battery capacity, stop-level charging maps, and load curves.

4. **PV Production Simulation**. Solar PV generation is estimated using environmental data and system parameters.  
*Main output*: Hourly PV production over a year.

5. **EV–PV Complementarity Analysis**. Assesses how well PV generation aligns with EV charging demand.  
*Main output*: Synergy metrics (e.g., self-sufficiency and self-consumption potentials) for the selected time period.



## Installation

### Requirements
- **Python**: Ensure Python is installed on your system. Note that the code was developed and tested using python 3.12, so other python version might not work.
- **Conda** (optional, but recommended): Use Conda for managing Python environments and dependencies. 

> :bulb: If you are new to python and conda environments, we recommand installing python and conda via the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution. During the installation, make sure to select "Add Miniconda to PATH" for ease of use.

> :thumbsdown: If you do not want to use conda, we strongly recommend using an other virtual environment manager (venv, ...). However, you can also manually install all the python dependencies (not recommended) using the list of required modules in the `environment.yml` file.

### Installation with conda
1. Clone the latest version of the code on GitHub on your local machine. If you are not familiar with git, you can also manually download the folder from GitHub and then run the code. However, you won't be able to contribute to the project.
```bash
$ git clone https://github.com/jeremydumoulin/evpv.git
```

2. Open an Anaconda prompt and create a new conda environment with the required dependencies. 
```bash
$ conda env create -f environment.yml -n your_environment_name
```

3. Activate the conda environment (assuming the environment is named `your_environment_name`). 
```bash
$ conda activate your_environment_name
```

4. Install evpv as a python package (optionnal, but recommended to streamline the import and execution of the code). The "-e" option allows changes to the model (e.g., in case of pull ) to be to reflected immediately without reinstallation.
```bash
$ pip install -e .
```

![](docs/installation.gif)

## Usage

The evpv-simulator model can be run in two ways:

1. **Basic Usage (command-line mode)**: Ideal for users who are not familiar with Python or who want to conduct a basic case study using a simple configuration file to specify the input parameters
2. **Advanced Usage**: Suitable for users who prefer to import and use the evpv-simulator model as python modules in their own scripts or need to conduct more advanced analyses (such as parametric studies, integrating the evpv-simulator with other Python packages, ...).

### Basic Usage

To run the evpv-simulator in command-line mode, follow these steps:

1. **Open a Terminal**: Use a terminal application, such as Anaconda Prompt.
2. **Activate the conda environment** 
3. **Create a configuration file for your case study**: Create a Python file (e.g., `config.py`) and populate it with your specific input values. Note that some input parameters are georeferenced data files (e.g., population raster). To see all required and optional parameters, refer to the `examples/Basic_AddisAbaba_ConfigFile/config.py` file. You can either clone this file to suit your needs or run it first to familiarize yourself with the inputs and outputs.
4. **Ensure that your terminal is opened in the directory where the config file is located, OR specify the absolute path to the input/output directories in the config file.**
5. **Run the EV-PV command-line script**: Execute the `evpv.py` script (depending on your Python installation, you may need to use `python3` instead of `python`):
   ```bash
   evpv
   ```
6. **Provide configuration file path**: When prompted, enter the path to your configuration file, such as the basic example:
   ```bash
   Enter the path to the python configuration file: C:\Users\(...)\config.py
   ```
7. **Check Outputs**: When to code is running, various output values will appear in the terminal. Once the simulation is complete, you will find all output files in the output directory you specified in your configuration file.

![](docs/usage.gif)

### Advanced Usage

For advanced users, you can also create a new Python script and manually import and interact with the classes in the `evpv/` folder (see the *Project Structure*). For a typical use, you will generally need to one ore more of the following classes:

```python
from evpv.vehicle import Vehicle
from evpv.vehiclefleet import VehicleFleet
from evpv.region import Region
from evpv.mobilitysimulator import MobilitySimulator
from evpv.pvsimulator import PVSimulator
from evpv.chargingsimulator import ChargingSimulator
from evpv.evpvsynergies import EVPVSynergies
```

> :information_source: Comprehensive documentation is still in progress. However, you can refer to the generated Sphinx documentation for detailed descriptions of the input and output parameters for each class (located in the `docs/_build/html/index.html` folder). For a quick start, we recommend exploring the example scripts in the `examples/` folder. We recommend starting with `01_basic_usage.py`, which demonstrates how to utilize the core classes with a minimal set of input parameters. 

## Project structure
```bash
├───environment.yml
├───setup.py
├───version.py
├───LICENSE.md
├───README.md
├───doc/
├───evpv/
│   ├───chargingsimulator.py
│   ├───evpvsynergies.py
│   ├───mobilitysimulator.py
│   ├───pvsimulator.py
│   ├───region.py
│   ├───vehicle.py
│   ├───vehiclefleet.py
│   ├───evpv_cli.py
│   └───helpers.py
├───examples/
│   ├───Basic_AddisAbaba_ConfigFile/
│   └───...
└───scripts/
```  
### evpv-simulator run script
The file `evpv_cli.py` is a python script that allows users to run to conduct a basic study using a simple command line interface (see section *Usage*).

### Python Modules
In the `evpv/` folder, you will find the following python classes:

**Core Classes**
These modules are essential for basic usage:
- **`Vehicle`**: Defines the parameters for a specific type of electric vehicle.
- **`VehicleFleet`**: Manages information about an EV fleet for simulation.
- **`Region`**: Represents the region of interest, including geospatial characteristics.
- **`MobilitySimulator`**: Simulates mobility demand by allocating EVs to origins and determining trip distributions.
- **`ChargingSimulator`**: Analyzes the spatial and temporal charging needs for the EV fleet.
- **`PVSimulator`**: Simulates the PV production based on PV-Lib.
- **`EVPVSynergies`**: Computes metrics for PV-based charging of EVs.

**Additional Files**
- **`evpv_cli.py`**: Provides a command-line interface for running mobility demand simulations.
- **`helpers.py`**: Contains various utility functions used internally by other classes.

### Examples
In the `examples/` folder, you will find various examples illustrating basic and more advanced use cases. We recommend looking at the various scripts, starting with the more basic ones.

### Scripts
In the `scripts/` folder, you will find additionnal helpful scripts, notably a script to fetch georeferenced workplaces or points of interest from OpenStreetMap.

## Features

### Main features
- **Endogenous estimation of daily mobility demand for home-to-work commuting.** To estimate charging needs in a specific area, it is essential to assess the commuting transport demand — specifically, the flow of vehicles between potential origin points (e.g., homes) and destination points (e.g., workplaces), as well as the road-based distance between them. The model estimates this demand internally by dividing the region of interest into traffic zones (based on user-defined spatial resolution) and applying a spatial interaction model to distribute the flow of people between their homes and workplaces (or other parking locations, such as park-and-ride facilities). A key feature of this model is its integration of the self-calibrated gravity model developed by [Lenormand et al.](https://doi.org/10.1016/j.jtrangeo.2015.12.008), which removes the need for transport data specific to the region. For accurate road distance calculations, the model utilizes OpenRouteService to perform routing when available in the region of interest.

- **Mobility demand for other purposes.** While this model primarily focuses on mobility demand for daily commuting, it may also be beneficial to incorporate mobility demand for other activities, such as shopping and leisure. Currently, this demand is not calculated endogenously within the model. However, users can include an optional parameter (called `km_per_capita_offset`) to account for these additional mobility needs on weekdays.

- **Spatial and temporal charging needs**. Based on the previously mentioned mobility demand, the model computes the daily spatial and temporal charging needs for electric vehicles. This is done using a scenario-based approach, where the user specifies the characteristics of the vehicle fleet (including the charging power for each vehicle) and the expected charging behaviors. For the spatial demand, the model calculates it for each traffic zone, based on the expected share of people charging either at home or at work. The temporal demand (charging curve) is estimated with a stochastic approach, where for each vehicle, several factors are randomly sampled: arrival time, daily travel distance and energy consumption (which determines the daily charging needs), and available charging power. Building on the work of [Pareschi et al.](https://doi.org/10.1016/j.apenergy.2020.115318), our model also introduces randomness in the number of vehicles that decide to charge on a given day. This decision is based on a threshold for the state of charge (SoC), with vehicles choosing to charge if their SoC falls below this threshold. 

- **Flexible configuration of EV fleet and charger power**. Supports diverse setups, including mixed charging power levels for each location, customizable vehicle types with user-defined maximum charging power for each vehicle.

- **Smart charging**: By default, the code simulates uncontrolled ("dumb") charging behavior. However, the data is pre-processed to allow easy post-analysis of smart charging strategies. A "peak shaving" algorithm is already implemented, enabling vehicles to adjust their charging patterns between arrival and departure times to smooth the overall charging demand curve. This smart charging behavior is managed using a simple rule-based algorithm.

- **PV power production.** The code offers a simple method to generate hourly photovoltaic (PV) production and other metrics, such as the capacity factor, for a specific location, year, and type of PV system (e.g., rooftop, ground-mounted, etc.). It is built on the PVLib library, which provides robust tools for simulating and analyzing PV system performance.
 
- **Potential for PV to cover the charging needs.** Using the charging curve and the local PV production, various metrics can be calculated to assess the potential of PV to meet the charging needs. These metrics include self-sufficiency potential, self-consumption, Spearman correlation, and more. The analysis can be performed for a specific day or over a longer time period.

### Limitations & Caveats

#### Mobility and EV Charging Demand

- **Zoning**: The spatial resolution is constrained by the size of the traffic zones, with no downscaling procedure available for the moment. The accuracy may also be affected when using a self-calibrated gravity model, which has not been validated for zones smaller than 5 km². Additionally, rectangular zoning may not be the optimal choice for transport demand modelling.
  
- **Trip purposes**: The mobility demand assumes direct trips between home and destination (and destination to home) with no intermediate stops. Alsom, other trip purposes than daily commuting cannot be modeled.

- **Routing**: Accurate routing depends on OpenRouteService, which requires an internet connection.

- **Weekdays only**: The model only accounts for mobility and charging on weekdays.

- **Zone attractiveness**: When using the number of workplaces from OpenStreetMap (OSM) to determine the attractiveness of a zone for trip distribution, it does not include the number of jobs per workplace. This may reduce the model’s accuracy in areas where the number of jobs per workplace is not evenly distributed.

- **Charging Curve**: The charging curve is calculated for each location and traffic zone, based on the zone-specific travel distance distribution and the assumed arrival time. The model assumes a normal distribution for arrival times, which may not fully represent real behavior. We also assume that all vehicles charge every day, which may not be entirely accurate (see [Pareschi et al.](https://doi.org/10.1016/j.apenergy.2020.115318)). Additionally, in reality, drivers may not charge their vehicles immediately upon arrival, as their decision could depend on factors such as varying electricity tariffs throughout the day (see, for instance, the charging habits of French citizen [here](https://www.enedis.fr/presse/mobilite-electrique-enedis-publie-deux-nouveaux-rapports-sur-les-habitudes-de-mobilite-et-de)). This behavior could be incorporated into a "smart charging" algorithm in future work to better reflect how drivers respond to price signals (or other incentives) when deciding when to charge.

#### Photovoltaic simulation

- **Weather Data**: The model relies on weather data from PVGIS, which requires an internet connection.

#### EV-PV complementarity

- **EV-PV System treated as a closed system**: When evaluating EV-PV synergies, we assume that all PV energy is available exclusively for EV charging. This approach ignores other potential loads and limitations associated with distributed PV systems, such as variations between different charging stations equipped with PV.

## Contributing
[To be completed]

### Open tasks

- [x] Update the README file to the new architecture
- [ ] Add more advanced examples
- [x] Enhance inline code documentation
- [x] Feature: energy demand per vehicle by traffic zone (i.e., use the distance distribution per zone and not the aggregated one). This will allow more advanced analysis, like variations in the average number of charging stations per vehicle because of higher travelled distances.
- [x] Create a CLI for easy usage
- [ ] Create a readthedocs
- [ ] Make a contributing guide
- [ ] Write some unit tests 

## Scientific publications
[1] Jérémy Dumoulin et al. A modeling framework to support the electrification of private transport in African cities: a case study of Addis Ababa. arXiv preprint arXiv:2503.03671, 2025. [https://doi.org/10.48550/arXiv.2503.03671](https://doi.org/10.48550/arXiv.2503.03671). 

## Acknowledgment 
This project was supported by the HORIZON [OpenMod4Africa](https://openmod4africa.eu/) project (Grant number 101118123), with funding from the European Union and the State Secretariat for Education, Research and Innovation (SERI) for the Swiss partners. We also gratefully acknowledge the support of OpenMod4Africa partners for their contributions and collaboration.

## License

[GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)
