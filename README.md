# GTFS4EV

**GTFS4EV (GTFS for Electric Vehicles) is an open-source Python tool designed to support strategic planning for the electrification of public transport systems. It simulates electric vehicle (EV) operations and charging scenarios based on the widely available GTFS (General Transit Feed Specification) data format. It enables users to explore and evaluate different electrification strategies, assess spatial and temporal charging needs, and determine the required battery capacities. It also integrates solar photovoltaic (PV) production modeling to analyze the complementarity between local renewable generation and EV charging needs.** 

In short, GTFS4EV helps answer key questions such as:
* What is the charging demand associated with the fleet electrification?
* Can a specific charging strategy meet this demand, and what battery capacity is required for the EVs?
* How can local solar PV generation be integrated to meet charging needs and reduce dependence on the electricity grid?
* What are the environmental benefits of electrification, such as reductions in CO₂ emissions, diesel fuel consumption, and reduction in exposure to air pollution?

Authors = Jeremy Dumoulin, Alejandro Pena-Bello, Noémie Jeannin, Nicolas Wyrsch

Lead institution = EPFL PV-LAB, Switzerland

Contact = jeremy.dumoulin@epfl.ch 

Langage = python 3 

> :bulb: This `README.md` provides a quick start guide for basic usage of the model. Comprehensive documentation for detailed and advanced usage will soon be available on a [Read the Docs](https://readthedocs.org/) page. 

## Table of Contents

1. [Overview of the Model](#overview-of-the-model)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Standout Features & Limitations](#standout-features--limitations)
6. [Scientific Publications](#scientific-publications)
7. [Acknowledgment](#acknowledgment)
8. [License](#license)

## Overview of the model
GTFS4EV follows a five-step workflow—from transit data preprocessing to the evaluation of synergies between electric vehicle (EV) charging and solar photovoltaic (PV) production. Below is a summary of the modeling steps and their typical outputs:

1. **GTFS Data Preprocessing**. GTFS data is loaded, checked, cleaned, and optionally filtered or enriched (e.g., by adding idle times at terminals).  
*Main outputs*: Cleaned GTFS data, transport network map.

2. **Fleet Operation Simulation**. Vehicle movements are simulated based on the GTFS feed, either for the entire network or selected trips.  
*Main output*: Operational data for each vehicle.

3. **Charging Scenario Simulation**. Charging needs are computed using customizable strategies and charging infrastructure setups.  
*Main outputs*: Charging schedules per vehicle, estimated battery capacity, stop-level charging maps, and load curves.

4. **PV Production Simulation**. Solar PV generation is estimated using environmental data and system parameters.  
*Main output*: Hourly PV production over a year.

5. **EV–PV Complementarity Analysis**. Assesses how well PV generation aligns with EV charging demand.  
*Main output*: Synergy metrics (e.g., self-sufficiency and self-consumption potentials) for the selected time period.

> :white_check_mark: Additional Python scripts are included to estimate avoided CO2 emissions, diesel fuel savings, and to map the reduction in population exposure to air pollution.

### Getting python
Ensure Python is installed on your system. This project was developped with **Python 3.12**. Other versions may not be compatible.

If it is your first time with Python, we recommend installing python via [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Many tutorials are available online to help you with this installation process (see for example [this one](https://www.youtube.com/watch?v=oHHbsMfyNR4)). During the installation, make sure to select "Add Miniconda to PATH".

> :thumbsup: Miniconda includes `conda`, which allows you to create a dedicated python environment for `gtfs4ev`. If not using conda, consider alternative environment managers like `venv`. Manual installation of all dependencies 
is also possible but not recommended.

### Installation 
1. (Optional) Create a Conda environment with Python 3.12. As stated before, it is not mandatory but recommended to use a dedicated environment. Here an example with conda using an environment named *gtfs4ev-env*

```bash
$ conda create --name gtfs4ev-env python=3.12
$ conda activate gtfs4ev-env
```

2. Install gtfs4ev as a python package from the GitHub repository
```bash
$ pip install git+https://github.com/jeremydumoulin/gtfs4ev.git
```

## Basic Usage

After installation, you can run the **GTFS4EV model in command-line mode**. This is ideal for users who are not familiar with Python or who want to quickly conduct a simple case study.

First, create a new configuration file for you case study by copying an existing example such as the `config.py` that you find in the `/example`. Update it with your own input values and make sure you have your GTFS data ready. GTFS data needs to be provided as a folder, not a .zip file. As for populating the input file, the `config.py` file comes with comments.

> :bulb: We recommend starting by running the example to get familiar with the workflow. The easiest way to access all necessary files is to download the full GitHub repository as a ZIP file, extract it and copy the contents of the example folder into the directory of your choice.

Once your config file and GTFS data foler is ready, open a terminal, activate your conda environment (optional), and run:
```bash
$ gtfs4ev
```
You’ll be prompted to enter the path to your config file:
```bash
$ Enter the path to the python configuration file: C:\Users\(...)\config.py
```
> :warning: Use absolute paths in the config file, or start the terminal in the same directory as the config file to use relative paths.

## Input Parameters

Input parameters are defined and explained in the configuration file. Some basic parameters (e.g., GTFS path, charging strategy) are mandatory, while advanced options (e.g., PV setup details) have default values and can be customized as needed.

> :bulb: Easily test different charging strategies or PV scenarios by editing the config file.

### Model Outputs

After simulation, outputs are organized into five folders:

- **GTFS**: Intermediate results related to the GTFS data, like aggregated information and a map of the transport network.
- **Mobility**: Intermediate results from the fleet operation simulation, such as the travel pattern of every vehicle, the distance travelled by the vehicles, and idle periods at stops and terminals.
- **Charging**: Contains results from the charging simulation, including: charging schedules per vehicle and per stop, estimated battery capacities, load curves 
- **PV**: Hourly solar PV generation over the simulation year, based on the defined system parameters and location-specific environmental data.
- **EVPV**: Analysis of complementarity between PV production and EV charging demand.

## Advanced usage

Advanced users can develop custom analyses or workflows by importing core classes from the `gtfs4ev/` module:

```python
from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev.fleetsimulator import FleetSimulator
from gtfs4ev.chargingsimulator import ChargingSimulator
# and more...
```

This allows full control over each step of the modeling process—from GTFS processing to charging simulation and result interpretation.

> :bulb: For a working example, see `run_script.py` in the `/example` folder.
> :bulb: Additional post-processing scripts (e.g., CO₂ savings, air pollution exposure analysis) are available in the `/scripts` folder.

## Standout Features 

### Main features
- **Fully Data-Driven Simulation**. Operates entirely from GTFS data (no need for additional operational datasets) to simulate electric bus operation and energy demand. Ideal for contexts where transport data is scarce.

- **Idle Time Handling**. Users can enrich GTFS data by adding customizable idle times at stops or terminals to better reflect real-world vehicle behavior.

- **Chained Charging Strategy Framework**. Supports multiple charging strategies applied in sequence. The model starts with the first strategy and falls back to the next one if charging needs are not met, ensuring flexible charging simulations.

- **Extensible Charging Strategy Design**. The codebase is modular and ready for other user-defined charging strategies, making it easy to plug in new logic without modifying the core simulation.

- **PV system presets**. Easily generates PV production and EV–PV complementarity metrics for common PV system types (rooftop, ground-mounted, with or without tracking).

### Planned features

- Adding more charging strategies
- Speed up calculations

## Scientific publications
(...)

## Acknowledgment 
This project was supported by the HORIZON [OpenMod4Africa](https://openmod4africa.eu/) project (Grant number 101118123), with funding from the European Union and the State Secretariat for Education, Research and Innovation (SERI) for the Swiss partners. We also gratefully acknowledge the support of OpenMod4Africa partners for their contributions and collaboration.

## License

[GNU GENERAL PUBLIC LICENSE](https://www.gnu.org/licenses/gpl-3.0.html)
