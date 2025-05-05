
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
4. [Features](#features)
5. [Project Structure](#project-structure)
6. [Scientific Publications](#scientific-publications)
7. [Acknowledgment](#acknowledgment)
8. [License](#license)

## Overview of the model
The model follows a five-step workflow from data preparation to the evaluation of solar-electric vehicle synergies. Below is a quick summary of the modelling steps and their typical outputs:

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

> :bulb: Additionally, the model provides Python scripts to easily assess CO₂ savings, fuel savings, and map the reduction in exposure to air pollution.

## Installation

### Requirements
- **Python**: Ensure Python is installed on your system. Note that the code was developed and tested using python 3.12, so other python version might not work.
- **Conda** (optional, but recommended): Use Conda for managing Python environments and dependencies. 

> :bulb: If you are new to python and conda environments, we recommand installing python and conda via the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution. During the installation, make sure to select "Add Miniconda to PATH" for ease of use.

> :thumbsdown: If you do not want to use conda, we strongly recommend using an other virtual environment manager (venv, ...). However, you can also manually install all the python dependencies (not recommended) using the list of required modules in the `environment.yml` file.

### Installation with conda


![](docs/installation.gif)

## Usage

### Basic Usage

### Advanced Usage

## Features

### Main features 

- **Lorem ipsum** Lorem ipsum dolor sit amet

### Available charging strategies

### Planned Features

- Update the README file to the new architecture
- Add more advanced examples
- Create a CLI for easy usage
- Create a readthedocs
- Make a contributing guide
- Write some unit tests 

## Project structure (for contributing or advanced usage)
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

## Scientific publications

## Acknowledgment 
This project was supported by the HORIZON [OpenMod4Africa](https://openmod4africa.eu/) project (Grant number 101118123), with funding from the European Union and the State Secretariat for Education, Research and Innovation (SERI) for the Swiss partners. We also gratefully acknowledge the support of OpenMod4Africa partners for their contributions and collaboration.

## License

[GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)
