#modification

## Prerequisites
1. [python 3.x](https://www.python.org/download/releases/3.0/) - core programming language for running CityChrone scripts.
2. [jupyter Notebook](http://jupyter.org/) -  interacting with the CityChrone analysis notebooks.
3. [MongoDB](https://www.mongodb.com/download-center#community) with the privileges to create and modified a database. CityChrone's backend relies on MongoDB for data storage and management. Ensure to dowload also MongoDB Compass - a GUI for MongoDB (to visually explore your data, build queries, and optimize databases).
4. [Docker](https://docs.docker.com/get-started/get-docker/) - to run applications inside containers. Make sure your system supports and has enabled hardware virtualization (usually found in BIOS/UEFI settings). WSL 2 Backend (Recommended): Windows Subsystem for Linux 2 should be enabled for better performance.
5. Install all the Python libraries listed below (see "Execution" section, Par. 1). Make sure to install the suggested versions of each library (functions you need might be removed or changed in newer versions).

## Installation
1. Clone this repository.
2. Run the commands annotated in `Docker-instructions.txt` 

You will see `running and waiting for requests` message, which means that OSRM server is ready

3. Install Mongo DB --> `sudo apt-get install mongodb`
4. Install IPython --> `pip install IPython`

## OSM DATA
OpenStreetMap (OSM) data provides detailed geographic information that CityChrone uses to compute accessibility.

Get OSM data for specific regions from [geofabrik](https://download.geofabrik.de/).

The .osm.pbf (Protocolbuffer Binary Format) is recommended as it is compact and efficient.

To compute the "Accessibility Score" the distribution of public services in the city is needed. These public services include bus stops, colleges, kindergartens, libraries, schools, research institutes, car-sharing points, clinics, doctors, dentists, pharmacies, veterinary services, social facilities, cinemas, community centres, social centres, theatres, market places, stop positions, platforms, stations, stop areas, and stop area groups. This data can be scraped from OpenStreetMap using the Python package OSMnx by running the notebook "scrapeOpportunitiesOSM.ipynp".

## Execution
1. Start Docker Desktop (once it's running, you should see a green light or a related message).

2. Run the following command from the "command prompt" (make sure you are inside the "osm" folder): 

	For Linux users (or Windows, using PoweShell):

		docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/Budapest.osrm

	For Windows users (using cmd):

		docker run -t -i -p 5000:5000 -v "%cd%:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/Budapest.osrm

This command starts the OSRM routing server and runs the routing engine, allowing you to make routing requests to the server (on port 5000).

3. Open the notebook "Accessibility_Analysis.ipynb" in you coding environment (e.g. VS code, Jupyter Notebook, ...). Install the packages below with the specific versions:

		geojson == 3.1.0
		pymongo == 4.7.2
		pandas == 2.1.4
		folium == 0.16.0
		numpy == 1.26.4
		numba == 0.59.0
		geopy == 2.4.1
		shapely == 2.0.4
		geopandas == 0.14.4


4. Set the variable listed at the start of the notebook:
	1. ```city = 'Budapest' # name of the city```
	2. ```urlMongoDb = "mongodb://localhost:27017/"; # url of the mongodb database```
	3. ```directoryGTFS = './gtfs/'+ city+ '/' # directory of the gtfs files.```
	4. ```day = "20170607" #hhhhmmdd [date validity of gtfs files]```
	5. ```dayName = "wednesday" #name of the corresponding day```
	6. ```urlServerOsrm = 'http://localhost:5000/'; #url of the osrm server of the city```
    \[\Optional -- population collection]
    7. ```urlMongoDbPop = "mongodb://localhost:27017/"; # url of the mongodb database of population data```
    8. ```popDbName = "" #name of the population database```
    9. ```popCollectionName = ""#name of the population collection```
    10. ```popField = ""#the field in the properties field in the elements containing the value of the population```
    11. ```set the reference system of the population file is in `EPSG:4326`. If your population files are in another reference system, you should first convert them, using some external tools (e.g., qGIS)```

5. run  all the cells in the notebook 

    





