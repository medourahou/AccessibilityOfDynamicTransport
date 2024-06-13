


## Prerequisites
1. [python 3.x](https://www.python.org/download/releases/3.0/)
1. [jupyter](http://jupyter.org/)
1. [MongoDB](https://www.mongodb.com/download-center#community) with the privileges to create and modified a database.
1. Docker
1. All the python library needed, listed at the beginning of the notebook.


## Installation
1. Clone this repository.
2. Run the commands annotated in `Docker-instructions.txt` 

You will see `running and waiting for requests` message, which means that OSRM server is ready

3. Install Mongo DB `sudo apt-get install mongodb`
4. Install `pip install IPython`


### Optional
To compute the "Sociality Score," the distribution of public services in the city is needed. These public services include bus stops, colleges, kindergartens, libraries, schools, research institutes, car-sharing points, clinics, doctors, dentists, pharmacies, veterinary services, social facilities, cinemas, community centres, social centres, theatres, market places, stop positions, platforms, stations, stop areas, and stop area groups. This data can be scraped from OpenStreetMap using the Python package OSMnx.

## Execution


1. Install the packages below with the specific versions:

		geojson == 3.1.0
		pymongo == 4.7.2
		pandas == 2.1.4
		folium == 0.16.0
		numpy == 1.26.4
		numba == 0.59.0
		geopy == 2.4.1
		shapely == 2.0.4
		geopandas == 0.14.4

2. Adjust the date indicated in the line `day = ...` so that it corresponds to a date that is contained in the GTFS file.

3. Ensure that the reference system of the population file is in `EPSG:4326`. If your population files are in another reference system, you should first convert them, using some external tools (e.g., qGIS).

4. Run the cells in the notebook `step_1_Estimation of travel time and wait time.ipynb` to make estimations at unsampled areas in the data and get new gtfs files with drt integrated.
5. Run the cells in the notebook `step_2_Compute the accessibilities.ipynb` to compute and visualize the accessibilities.

    





