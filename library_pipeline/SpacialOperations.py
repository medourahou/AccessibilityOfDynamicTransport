import logging
import math
from shapely.geometry import Polygon
import shapefile as shp
import pyproj
import tqdm
import pandas as pd
import geopandas as gpd
import logging
import numpy as np

def runGridCreation(studyarea_input_path, gridSize):
    bBox = extractBoundingBoxFromShp(studyarea_input_path)
    polygons, centroids, Centroids_X , Centroids_Y = createSpatialGrid(bBox, gridSize)
    CellsDF = createGeoDataFrame(polygons, 'EPSG:3035', pd.DataFrame())
    CellsDF['Centroids'] = centroids
    CellsDF['Centroid_X'] = Centroids_X
    CellsDF['Centroid_Y'] = Centroids_Y
    CellsDF = removeExcessGrid(CellsDF, studyarea_input_path, 3035)
    return CellsDF

def extractBoundingBoxFromShp(shapeFilePath):
    proj = pyproj.Transformer.from_crs(4326, 3035, always_xy=True)
    shape = shp.Reader(shapeFilePath)
    raw = shape.bbox
    x1, y1 = proj.transform(raw[0], raw[1])
    x2, y2 = proj.transform(raw[2], raw[3])
    return [int(x1),int(y1),int(x2),int(y2)]

def extractBoundingBoxFromPointCloud(df):
    xMin = df['start_x'].min()
    xMax = df['start_x'].max()
    yMin = df['start_y'].min()
    yMax = df['start_y'].max()
    return[int(xMin),int(yMin), int(xMax), int(yMax)]

def getRadiusofPointCloud(df):
    bbox = extractBoundingBoxFromPointCloud(df)
    diagonal = math.sqrt(math.pow(bbox[2]-bbox[0],2) + math.pow(bbox[1]-bbox[3],2))
    radius = diagonal/2 * (2/3)
    return radius

def pointinCircle(radius,center,point):
    dist = math.sqrt(math.pow(center[0]-point[0],2) + math.pow(center[1]-point[1],2))
    return dist <= radius


def getHexagonCenterpoints(bbox, cellSize):
    xmin, ymin, xmax, ymax = bbox
    xmid = xmin + (xmax - xmin) / 2
    ymid = ymin + (ymax - ymin) / 2
    X = []
    xVals = [xmid]
    for xdif in range(0, int((xmax - xmin) / 2), cellSize):
        xVals.append(xVals[-1] + cellSize)
        xVals.insert(0, xVals[0] - cellSize)
    X.append(xVals)
    xVals = []
    for xdif in range(0, int((xmax - xmin) / 2), cellSize):
        if not xVals:  # check if list is empty
            xVals.append(xmid + cellSize / 2)
            xVals.insert(0, xmid - cellSize / 2)
        else:
            xVals.append(xVals[-1] + cellSize)
            xVals.insert(0, xVals[0] - cellSize)
    X.append(xVals)
    Y = [ymid]
    deltaY = int(round((1.5*cellSize)/math.sqrt(3),0))
    for ydif in range(0, int((ymax - ymin) / 2), deltaY):
        Y.append(Y[-1] + deltaY)
        Y.insert(0, Y[0] - deltaY)
    return X,Y

def constructHexagonWKT(x,y,r):
    R = (r*2)/(math.sqrt(3))
    X = [x+0,x+r,x+r,x+0,x-r,x-r]
    Y = [y+R,y+R/2,y-R/2,y-R,y-R/2,y+R/2]
    wkt = 'POLYGON (('
    for i in range(0,6,1):
        wkt = wkt + str(int(round(X[i],0))) + ' ' + str(int(round(Y[i],0))) + ', '
    wkt = wkt + str(int(round(X[0],0))) + ' ' + str(int(round(Y[0],0))) + '))'
    return wkt

def createSpatialGrid(bbox, cellSize):
    X, Y = getHexagonCenterpoints(bbox, cellSize)
    Polygons = []
    Centroids = []
    Centroids_X = []
    Centroids_Y = []
    progress = tqdm.tqdm(total=((len(Y)-1)/2), desc='Progress', position=0)
    for iY in range(0,len(Y)-1,2):
        progress.update(1)
        y1 = Y[iY]
        y2 = Y[iY+1]
        for iX in range(0,int(min(len(X[0]),len(X[1]))),1):
            x1 = X[0][iX]
            x2 = X[1][iX]
            r = cellSize/2
            h1 = constructHexagonWKT(x1, y1, r)
            h2 = constructHexagonWKT(x2, y2, r)
            p1 = 'POINT ('+str(x1)+' '+str(y1)+')'
            p2 = 'POINT (' + str(x2) + ' ' + str(y2) + ')'
            Centroids.append(p1)
            Centroids.append(p2)
            Polygons.append(h1)
            Polygons.append(h2)
            Centroids_X.append(x1)
            Centroids_X.append(x2)
            Centroids_Y.append(y1)
            Centroids_Y.append(y2)
    return Polygons, Centroids , Centroids_X, Centroids_Y

def createGeoDataFrame(geodata, crs, df):
    gs = gpd.GeoSeries.from_wkt(geodata,crs=crs)
    return gpd.GeoDataFrame(df, geometry=gs, crs=crs)

def removeExcessGrid(gdf, shp_studyArea, crs, shp_popArea, cutPopArea):
    logging.info('Removing hexagons outside of the regarded study area')
    df = gpd.read_file(shp_studyArea)
    df = df.to_crs(epsg=crs)
    studyArea = df['geometry'][0]
    polygon = Polygon(studyArea)
    gdf['inStudyArea'] = gdf['geometry'].overlaps(polygon) | gdf['geometry'].within(polygon)
    gdf = gdf[gdf['inStudyArea'] == True]
    #gdf.insert(loc=0,column='cell_id',value=range(0, len(gdf)))
    #gdf.set_index('cell_id')
    if cutPopArea:
        logging.info('Removing hexagons outside of populated area')
        df = gpd.read_file(shp_popArea)
        df = df.to_crs(epsg=crs)
        popArea = df['geometry'][0]
        polygon = Polygon(popArea)
        gdf['inStudyArea'] = gdf['geometry'].overlaps(polygon) | gdf['geometry'].within(polygon)
        gdf = gdf[gdf['inStudyArea'] == True]
    gdf.insert(loc=0, column='cell_id', value=range(0, len(gdf)))
    gdf.set_index('cell_id')

    return gdf

def countTripsPerCell(cellGdf,tripGdf):
    logging.info('Counting trips originating in each cell of hexagonal grid')
    workingDF = gpd.sjoin(tripGdf, cellGdf, how='inner', predicate='intersects')
    counts = workingDF.groupby(['cell_id'], as_index=False, )['agentID'].count()
    counts.columns = ['cell_id', 'origin_count']
    cellGdf = cellGdf.merge(counts, on='cell_id', how='left')
    cellGdf['origin_count'].fillna(0, inplace=True)
    return cellGdf

def addCellId2Trip(cellGdf,tripGdf):
    logging.info('Assigning corresponding cell to origin of trips')
    workingDF = gpd.sjoin(tripGdf, cellGdf, how='inner', predicate='intersects')
    tripGdf['originCell'] = workingDF['cell_id']

def aggregateTripsByCell(tripGdf,cellGdf,hubsDF):
    workingDF = tripGdf[['originCell','<leg>_trav_time','Hub']]
    workingDF = workingDF.groupby(['originCell','Hub'],as_index=False)['<leg>_trav_time'].mean()
    workingDF = pd.merge(cellGdf, workingDF, how='left', left_on='cell_id', right_on='originCell', validate='1:m')
    workingDF.sort_values(by='cell_id').reset_index(inplace=True)
    workingDF = pd.merge(workingDF,hubsDF[['position', 'name']],how='left',left_on='Hub',right_on='name')
    workingDF.drop(columns=['name', 'inStudyArea','originCell'], inplace=True)
    workingDF['position'] = workingDF['position'].fillna(0)
    return workingDF

def Points2Line(df,col1,col2):
    routes = []
    for i, row in df.iterrows():
        if row[col2] != 0:
            p1 = row[col1].strip('POINT (')
            p1 = p1.strip(')')
            p2 = row[col2].strip('POINT (').strip(')')
            p2 = p2.strip(')')
            r = 'LINESTRING (' + p1 + ', ' + p2 +')'
            routes.append(r)
        else:
            routes.append('')
    df['Routes'] = routes
