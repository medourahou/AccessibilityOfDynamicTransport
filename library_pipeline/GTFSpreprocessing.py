
import sys
sys.path.insert(0, '../library/')
import pymongo as pym
from pathlib import Path
import geopandas as gpd
import geojson
from IPython.core.display import display, HTML


# This reduces GTFSpreprocessing executes the first steps of the public-transport-analysis tool
# The output of the run(outputPath) function is a csv file containing the hexagonal zones within the bounding box
# defined by the extent of the stops given in the gtfs file.
#
# The GTFS file is read and put into the mongodb database
# Stops are analysed and the bounding box is calculated
# The hexagonal grid is calculated
#
# Required Data and Preparation:
# -mongoDB server has to be set up and the address needs to be provided
# -GTFS file (.zip) of the respected area needs to be provided
#  -->a date for has to be chosen within the valid days of the GTFS file
# -Shapefile (.shp) including population data for study area (e.g. spacial grid)
#  -->name of the filed in shapefile holding the pop data
#

def run(outputPath):
    # general info
    city = 'Paris'

    # Study area
    # Study area defines the area in which the accessibility calaculation is executes
    study_area_shp_path = 'Paris/shp/cps.shp'

    # file paths and addresses to be provided:
    # population data
    shpPath = 'Paris/shp/ParisPoP.shp'
    directoryGTFS = './Paris/gtfs/basecase/' # !! only directory of gtfs.zip file, not the path to file
    urlMongoDb = "mongodb://localhost:27017/";  # url of the mongodb database

    # required parameters
    # regarded day from gtfs file
    day = '20220307'
    dayName = "monday"  # name of the corresponding day
    # parameters of walking distance
    timeWalk = 15 * 60  # seconds
    velocityWalk = 1.39  # m/s ***5km/h***
    distanceS = timeWalk * velocityWalk
    # Parameters thst define the resolution and extention of tesselletion and the maximum of the walking time
    # grid step of the hexagonal tesselletion in kilometers
    gridEdge = 1

    # Set check4stops = False if cells / hexagones should be included that do not have stops within.
    # Set check4stops = False for preprocessing prior to dynamic mode to gtfs convertion
    # Set check4stops = True for citychrone accessibility analysis
    check4stops = False


    # NO MORE CHANGES REQUIRED BELOW THIS LINE

    client = pym.MongoClient(urlMongoDb)
    gtfsDB = client['PublicTransportAnalysis']

    
    import printGtfsDate
    printGtfsDate(directoryGTFS)

    #  Start of the computation
    #  Read stops, routes, trips, calendar and calendar_dates from gtfs
    #  add population data

    geojsonPath = shpPath.split('.')[0] + '.geojson'
    shapefile = gpd.read_file(Path(shpPath))
    shapefile.to_file(Path(geojsonPath), driver='GeoJSON')

    with open(Path(geojsonPath)) as f:
        gj = geojson.load(f)
    features = gj['features']
    gtfsDB["POP"].drop()
    gtfsDB["POP"].insert_many(features)


    from library.libStopsPoints import loadGtfsFile
    listOfFile = ['stops.txt', 'routes.txt', 'trips.txt', 'calendar.txt', 'calendar_dates.txt',
                  'stop_times.txt']  #'shapes.txt']
    loadGtfsFile(gtfsDB, directoryGTFS, city, listOfFile)

    #  Fill the database with the connections
    from library.libConnections import readConnections
    readConnections(gtfsDB, city, directoryGTFS, day, dayName)

    # remove stops with no connections
    # and add to each stop the pos field
    from library.libStopsPoints import removingStopsNoConnections, setPosField, removeStopsOutBorder
    # removeStopsOutBorder(gtfsDB, city, 'OECD_city', ["commuting_zone", "city_core"])
    removingStopsNoConnections(gtfsDB, city)
    setPosField(gtfsDB, city)


    from library.libConnections import updateConnectionsStopName
    updateConnectionsStopName(gtfsDB, city)
    # Tassel with exagons
    #  List of all stops

    from library.libStopsPoints import returnStopsList
    stopsList = returnStopsList(gtfsDB, city)


    if len(study_area_shp_path) != 0:
        print('Study area from Shapefile')
        from library.libStudyArea import extractBoundingBoxFromShp
        bbox = extractBoundingBoxFromShp(study_area_shp_path)
    else:
        print('Study area from gtfs stop extend')
        from library.libStopsPoints import boundingBoxStops, mapStops
        from IPython.core.display import display, HTML
        display(HTML('<h1>All stops of the public transport present in the gtfs files</h1>'))
        bbox = boundingBoxStops(stopsList)
        mapStops(bbox, stopsList)

    #  Tassel the box with exagons.
    from library.libHex import hexagonalGrid
    hexBin, pointBin = hexagonalGrid(bbox, gridEdge, gtfsDB['stops'], distanceS, city,check4stops)

    # exporting hexagonal grid to csv file
    import csv
    field_names = ["point","hex","city","served","pos"]

    with open(outputPath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=';')
        writer.writeheader()
        writer.writerows(pointBin)

