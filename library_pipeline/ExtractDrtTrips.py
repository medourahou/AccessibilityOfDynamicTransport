import logging
import math
import pandas as pd
import tqdm
import pyproj

def ensureImportTypeConsistency(df):
    logging.info("Importing file and checking for valid Datatypes")
    cols = list(df.columns)
    dfOut = pd.DataFrame(columns=cols)
    progress = tqdm.tqdm(total=len(df), desc='Progress', position=0)
    for index, row in df.iterrows():
        progress.update(1)
        rowData = []
        for i in range(len(row)):
            if isinstance(row[i], str):
                if row[i][0] == "[" and row[i][1] == "'":
                    input = row[i]
                    input = input.strip("][")
                    input = input.split(", ")
                    for i, x in enumerate(input):
                        if len(x) <= 2:
                            input[i] = ""
                        else:
                            input[i] = input[i].strip("'")
                    rowData.append(input)
                else:
                    rowData.append(row[i])
            else:
                rowData.append(row[i])
        dfOut.loc[len(dfOut)] = rowData
    return dfOut


def run(dfIn,colsInput):
    logging.info('Extracting DRT legs:')
    # columns in DrtTrip dataframe need to be named like in the plans dataframe

    colsOutput = ['agentID', '<leg>_dep_time', '<leg>_trav_time', 'start_x', 'start_y', 'end_x',
                 'end_y', '<route>_directRideTime', '<route>_distance']

    if len(colsInput) == 0:
        colsInput = ['agentID', '<leg>_dep_time', '<leg>_trav_time', '<activity>_x', '<activity>_y', '<activity>_x',
                 '<activity>_y', '<route>_directRideTime', '<route>_distance']

    dfOut = pd.DataFrame(columns=colsOutput)
    progress = tqdm.tqdm(total=len(dfIn), desc='Progress', position=0)
    for index, row in dfIn.iterrows():
        legModes = row["<leg>_mode"]
        #if isinstance(legModes, str):
        #    legModes = legModes.strip('][').split(', ')
        for i, mode in enumerate(legModes):
            data = []
            if mode == 'drt':
                data.append(convertValues(row[colsInput[0]]))
                data.append(convertValues(row[colsInput[1]][i]))
                data.append(convertValues(row[colsInput[2]][i]))
                data.append(convertValues(row[colsInput[3]][i]))
                data.append(convertValues(row[colsInput[4]][i]))
                data.append(convertValues(row[colsInput[5]][i+1]))
                data.append(convertValues(row[colsInput[6]][i+1]))
                data.append(convertValues(row[colsInput[7]][i]))
                data.append(convertValues(row[colsInput[8]][i]))
                dfOut.loc[len(dfOut)] = data
        progress.update(1)
    return dfOut

def convertValues(value):
    if ':' in value:
        a = value.find(":")+1
        b = value.find(":",a)+1
        h = (value[0:2])
        m = (value[a:a+2])
        s = (value[b:b+2])
        return int(s) + int(m) * 60 + int(h) * 3600
    elif '.' in value:
        return round(float(value),2)
    else:
        return value

def calcBLinedistances(df):
    logging.info('Calculating B-Line distance')
    distances = []
    for index, row in df.iterrows():
            x1 = row['start_x']
            x2 = row['end_x']
            y1 = row['start_x']
            y2 = row['end_x']
            dist = math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
            distances.append(dist)
    df['B-line_distance'] = distances

def calcRideTimeFactor(df):
    logging.info('Calculating ride time factor')
    values = []
    for index, row in df.iterrows():
        a = row['<leg>_trav_time']
        b = row['<route>_directRideTime']
        val = a/b
        values.append(val)
    df['rideTimeFactor'] = values

def calcRideDistanceFactor(df):
    logging.info('Calculating ride distance factor')
    logging.warning('b-line distance is used!')
    values = []
    for index, row in df.iterrows():
        a = row['<route>_distance']
        b = row['B-line_distance']
        val = a/b
        values.append(val)
    df['rideDistanceFactor'] = values

def constructWktPoint(df, xCol, yCol, newColName):
    logging.info('Converting Coordinates to WKT')
    df[newColName] = None
    for index, row in df.iterrows():
        x = row[xCol]
        y = row[yCol]
        wkt = "POINT (" + str(x) + " " + str(y) + ")"
        df.loc[index,newColName] = wkt
    #df.drop(columns= [xCol,yCol], axis=1, inplace=True)

def constructWktODline(df, xO, yO, xD, yD, newColName):
    logging.info('Creating OD lines')
    df[newColName] = None
    for index, row in df.iterrows():
        x1 = row[xO]
        y1 = row[yO]
        x2 = row[xD]
        y2 = row[yD]
        wkt = "LINESTRING (" + str(x1) + " " + str(y1) +", " + str(x2) + " " + str(y2) + ")"
        df.loc[index, newColName] = wkt
    #df.drop(columns=[xO, yO, xD, yD], axis=1, inplace=True)

def convertCRS(df,from_crs, to_crs,xO, yO, xD, yD):
    logging.info('Ensuring correct CRS')
    proj = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for index, row in df.iterrows():
        x1, y1 = proj.transform(row[xO], row[yO])
        x2, y2 = proj.transform(row[xD], row[yD])
        df.loc[index, xO] = x1
        df.loc[index, yO] = y1
        df.loc[index, xD] = x2
        df.loc[index, yD] = y2

def timeFrame2intervals(timeFrame, deltaT):
    start = convertValues(timeFrame[0])
    end = convertValues(timeFrame[1])
    step = deltaT * 60 * 60
    cuts = list(range(start, end, step))
    cuts.append(end)
    return cuts