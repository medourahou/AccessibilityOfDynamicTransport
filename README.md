[![DOI](https://zenodo.org/badge/104480172.svg)](https://zenodo.org/badge/latestdoi/104480172)
# public-transport-analysis
Urban Public transport analysis (POLITO-TSP).
This repository contains a jupyter notebook and all the related libraries to perform some of the analysis shown  in the <a href="http://citychrone.org" target="_blank">CityChrone platform</a> and compute the data nedeed to add new city in the CityChrone platform.


## Prerequisites
1. [python 3.x](https://www.python.org/download/releases/3.0/)
1. [jupyter](http://jupyter.org/)
1. [MongoDB](https://www.mongodb.com/download-center#community) with the privileges to create and modified a database.
1. Docker
1. All the python library needed, listed at the beginning of the notebook.


## Installation
1. Clone this repository.
2. Go to `public-transport-analysis/osrm` and run the commands annotated in `Docker-instructions.txt` 

You will see `running and waiting for requests` message, which means that OSRM server is ready

3. Install Mongo DB `sudo apt-get install mongodb`
4. Install `pip install IPython`


### Optional
To compute the "Sociality Score," the distribution of public services in the city is needed. These public services include bus stops, colleges, kindergartens, libraries, schools, research institutes, car-sharing points, clinics, doctors, dentists, pharmacies, veterinary services, social facilities, cinemas, community centres, social centres, theatres, market places, stop positions, platforms, stations, stop areas, and stop area groups. This data can be scraped from OpenStreetMap using the Python package OSMnx.

## Execution

## First run for Budapest

1. Open file `public-transport-city.py` and install the libraries that are imported at the beginning (use python3!)
     ATTENTION: You need to use specific versions of the following libraries:

		a.	pymongo   V 3.12.1

		b.	pandas    V 1.5.3

		c.	folium    V 0.14.0 

		d.	numpy     V 1.24.3 

		e.	requests  V 2.29.0

		f.	numba     V 0.57.0

		h.	geopy     V 2.4.1

		i.	shapely   V 1.8.0

		j.	datetime  V 5.4

2. Adjust the date indicated in the line `day = ...` so that it corresponds to a date that is contained in the GTFS file.

3. Ensure that the reference system of the population file is in `EPSG:4326`. If your population files are in another reference system, you should first convert them, using some external tools (e.g., qGIS).

4. To compute the accessibility scores, run `python3 public-transport-city.py`. Inside that script there is a variable `first_run`. By default it is True, which implies the mongo db is modified (adding links, connections, nodes, etc.). However, if you have already filled the databse, e.g., you are running the script for a second time, you do not need to fill the database again: in this case, set ``first_run=False` before running the script.

5. Results are written in the mongo-db, in the table `points`, where fields concerning sociality and velocity score are added

6. Use `public-transport-city.ipnb` to visualize the accessibility map
    
## Compute travel time distances and all the accessbility quantities
1. run ```jupyter-notebook``` and open the public-transport-analysis notebook.
1. Set the variable listed at the start of the notebook:
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
1. run the cells in the notebook.




