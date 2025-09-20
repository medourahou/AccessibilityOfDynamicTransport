import pandas as pd
import numpy as np
import pyproj
import geopandas as gpd
import logging

def importcells_run(path, w_crs):
    logging.info('Importing hexagonal grid created by Public-Transport-Analysis tool')
    Xpoints, Ypoints, Xhex, Yhex = importCellsCSV(path)
    df = createCellsDF(Xpoints, Ypoints, Xhex, Yhex, w_crs)
    return df

def importCellsCSV(path):
    df = pd.read_csv(path, sep=',', header=0)
    Xpoints = []
    Ypoints = []
    Xhex = []
    Yhex = []
    
    for i, row in df.iterrows():
        # Parse point coordinates
        pointStr = row['point']
        i1 = (pointStr.find('[')) + 1
        i2 = (pointStr.find(']'))
        pointStr = pointStr[i1:i2]
        point = pointStr.split(', ')
        Xpoints.append(float(point[0]))
        Ypoints.append(float(point[1]))
        
        # Parse polygon coordinates
        polygonStr = row['hex']
        i1 = (polygonStr.find('(((')) + 3
        i2 = (polygonStr.find('))'))
        polygonStr = polygonStr[i1:i2]
        # Split the string into coordinate pairs
        coords = polygonStr.strip('()').split('), (')
        for coord_pair in coords:
            x, y = map(float, coord_pair.split(', '))
            Xhex.append(x)
            Yhex.append(y)
            
    return Xpoints, Ypoints, Xhex, Yhex

def createCellsDF(Xpoints, Ypoints, Xhex, Yhex, w_crs):
    Points = []
    Xproj = []
    Yproj = []
    Hexs = []
    proj = pyproj.Transformer.from_crs(4326, w_crs, always_xy=True)
    
    # Create centroids
    for i in range(len(Xpoints)):
        x, y = proj.transform(Xpoints[i], Ypoints[i])
        Points.append(f'POINT ({x} {y})')
        Xproj.append(x)
        Yproj.append(y)
    
    # Create hexagon polygons with full precision
    for i in range(0, len(Xhex), 7):
        wkt = 'POLYGON (('
        coords = []
        for j in range(7):  # Include all 7 points to close the polygon
            x, y = proj.transform(Xhex[i+j], Yhex[i+j])
            coords.append(f'{x} {y}')
        wkt += ', '.join(coords) + '))'
        Hexs.append(wkt)
    
    # Create GeoDataFrame
    gs = gpd.GeoSeries.from_wkt(Hexs, crs=f'EPSG:{w_crs}')
    CellsDF = gpd.GeoDataFrame(pd.DataFrame(), geometry=gs, crs=f'EPSG:{w_crs}')
    CellsDF['Centroids'] = Points
    CellsDF['Centroid_X'] = Xproj
    CellsDF['Centroid_Y'] = Yproj
    
    return CellsDF