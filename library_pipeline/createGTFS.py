import pandas as pd
import pyproj
import tqdm
import logging
from library_pipeline import ExtractDrtTrips as edt
import random

def run(cellsDF, hubsDF, deltaT, timeFrame, crs_working):
    gtfs = createGTFSdataframes()
    addStops2DF(cellsDF, hubsDF, gtfs, crs_working)
    addStopTimes2DF(cellsDF, gtfs, hubsDF, deltaT, timeFrame) # --> adapt to accommodate multi interval analysis
    addTrips2DF(gtfs)
    addCalendar2DF(gtfs)
    addRoutes2DF(gtfs)
    addAgency2DF(gtfs)
    return gtfs

def createGTFSdataframes():
    #create all files that are neccessary for GTFS format als pandas DataFrames
    agency = pd.DataFrame(columns=['agency_id','agency_name','agency_url','agency_timezone'])
    calendar_dates = pd.DataFrame(columns=['service_id','date','exception_type'])
    calendar = pd.DataFrame(columns=['service_id','monday','tuesday','wednesday','thursday','friday','saturday','sunday','start_date','end_date'])
    feed_info = pd.DataFrame(columns=['feed_publisher_name','feed_publisher_url','feed_lang','feed_start_date','feed_end_date','feed_version'])
    routes = pd.DataFrame(columns=['route_id','agency_id','route_short_name','route_type'])
    stop_times = pd.DataFrame(columns=['trip_id','arrival_time','departure_time','stop_id','stop_sequence','pickup_type','drop_off_type','timepoint'])
    stops = pd.DataFrame(columns=['stop_id','stop_name','stop_lat','stop_lon','location_type','parent_station','platform_code','stop_url'])
    trips = pd.DataFrame(columns=['route_id','service_id','trip_id','trip_headsign'])

    gtfs = {
        "agency": agency,
        "calendar_dates": calendar_dates,
        "calendar": calendar,
        "feed_info": feed_info,
        "routes": routes,
        "stop_times": stop_times,
        "stops": stops,
        "trips": trips
    }
    return gtfs

def addStops2DF(cellsDF, hubsDF, gtfs, crs_working):
    proj = pyproj.Transformer.from_crs(crs_working, 4326, always_xy=True)
    stops = gtfs['stops']
    # stops in cells
    uniqueCells = cellsDF.drop_duplicates(subset=['cell_id'])
    progress = tqdm.tqdm(total=(len(uniqueCells['cell_id']) + len(hubsDF['name'])), desc='adding all stops to gtfs: stops.txt', position=0)
    print('Creating stops in cells')
    for i, row in uniqueCells.iterrows():
        j = len(stops)
        stops.loc[j, 'stop_id'] = row['cell_id']
        stops.loc[j, 'stop_name'] = str('virtual_' + str(row['cell_id']))

        x, y = proj.transform(row['Centroid_X'], row['Centroid_Y'])

        stops.loc[j, 'stop_lon'] = x
        stops.loc[j, 'stop_lat'] = y

        progress.update(1)
    #stops at hubs
    print('Creating stops at hubs')
    for i, row in hubsDF.iterrows():
        j = len(stops)

        stops.loc[j, 'stop_id'] = row['name']
        stops.loc[j, 'stop_name'] = row['name']

        wkt = row['position']
        wkt_parts = wkt.split(' ')
        wkt_parts[1] = wkt_parts[1].strip('(')
        wkt_parts[2] = wkt_parts[2].strip(')')

        x, y = proj.transform(float(wkt_parts[1]), float(wkt_parts[2]))

        stops.loc[j, 'stop_lon'] = x
        stops.loc[j, 'stop_lat'] = y

        progress.update(1)


def addStopTimes2DF(cellsDF, gtfs, hubsDF, deltaT, timeFrame):
    stop_times = gtfs['stop_times']
    stops = gtfs['stops'].copy()
    cellsCLEAN = pd.DataFrame(columns=cellsDF.columns)
    #cellsDF = cellsDF.dropna()
    cellsDF = cellsDF.fillna(-999)
    # intervalsALL contains all intervals defined by start, end and time step
    intervalsALL = edt.timeFrame2intervals(timeFrame, deltaT)
    progress = tqdm.tqdm(total=len(stops['stop_id']), desc='adding all waiting and travel times to gtfs: stop_times.txt', position=0)
    for stop in stops['stop_id']: # using cell_id = stop_id
        if 'Hub' in str(stop):
            progress.update(1)
            continue
        # cell is df with entries only of one specific cell for which a stop exists
        cell = cellsDF[cellsDF['cell_id'] == stop]
        #cell = fillMissingIntervals(cell, intervalsALL, len(hubsDF))
        cell = fillMissingIntervals_2(cell, intervalsALL, len(hubsDF), deltaT)
        cellsCLEAN = pd.concat([cellsCLEAN,cell], ignore_index=True)
        cellCopy = cell.copy()
        #stop_times = fillStopTimeTable(intervalsALL,stop_times,cell,len(hubsDF))
        fillStopTimeTable(intervalsALL, stop_times, cell, len(hubsDF),'access')
        fillStopTimeTable(intervalsALL, stop_times, cellCopy, len(hubsDF), 'egress')
        progress.update(1)
    #cellsCLEAN.to_csv('output/cellsCLEAN.csv', sep=';')



def addTrips2DF(gtfs):
    logging.info('adding trips to gtfs: trips.txt')
    trips = gtfs['trips']
    stop_times = gtfs['stop_times'].copy()
    stop_times.drop_duplicates(subset=['trip_id'], inplace=True)
    stop_times.reset_index(inplace=True)
    trip_id = stop_times['trip_id']
    route_id = []
    for x in trip_id:
         route_id.append(x.split('>')[0])
    trips['route_id'] = route_id
    trips['trip_id'] = trip_id
    for i, row in trips.iterrows():
        trips.loc[i,'service_id'] = 'all'

def addCalendar2DF(gtfs):
    logging.info('adding operation dates to gtfs: calendar.txt')
    service_id = gtfs['trips'].copy()
    service_id.drop_duplicates(subset=['service_id'], inplace=True)
    service_id.reset_index(inplace=True)
    s_id = service_id['service_id']
    calendar = gtfs['calendar']
    cols = ['service_id', 'start_date', 'end_date', 'monday','tuesday','wednesday','thursday', 'friday', 'saturday', 'sunday']
    data = [['','19990101','21003112',1,1,1,1,1,1,1]]
    calRow = pd.DataFrame(data, columns=cols)
    for s in s_id:
        calRow.loc[0,'service_id'] = s
        calendar = pd.concat([calendar,calRow], ignore_index=True)
    gtfs['calendar'] = calendar
    logging.info('Assuming same operation on every day of the week')

def addRoutes2DF(gtfs):
    logging.info('adding routes gtfs: routes.txt')
    trips = gtfs['trips'].copy()
    trips.drop_duplicates(subset=['route_id'], inplace=True)
    trips.reset_index(inplace=True)
    route_id = gtfs['routes']
    cols = ['route_id','agency_id','route_short_name','route_type']
    data = [['','DRT_virtual_service','',1]]
    # route_type = 1 --> suburban rail, every suburban transport system
    Row = pd.DataFrame(data, columns=cols)
    for r in trips['route_id']:
        Row.loc[0,'route_id'] = r
        Row.loc[0, 'route_short_name'] = 'DRT_' + r
        route_id = pd.concat([route_id,Row], ignore_index=True)
    gtfs['routes'] = route_id

def addAgency2DF(gtfs):
    logging.info('adding agencies to gtfs: agency.txt')
    routes = gtfs['routes'].copy()
    routes.drop_duplicates(subset=['agency_id'], inplace=True)
    routes.reset_index(inplace=True)

    agency = gtfs['agency']
    cols = ['agency_id','agency_name','agency_url','agency_timezone']
    data = [['','DRT_virtual_service','https://tum.de','Europe/Berlin']]
    Row = pd.DataFrame(data, columns=cols)

    for a in routes['agency_id']:
        Row.loc[0, 'agency_id'] = a
        agency = pd.concat([agency, Row], ignore_index=True)
    gtfs['agency'] = agency

def sec2str(seconds):
    if int(seconds/3600) < 10:
        h = '0' + str(int(seconds/3600))
    else:
        h = str(int(seconds/3600))
    seconds = seconds % 3600
    if seconds/60 < 10:
        m = '0' + str(int(seconds / 60))
    else:
        m = str(int(seconds / 60))
    seconds = seconds % 60
    if seconds < 10:
        s = '0' + str(seconds)
    else:
        s = str(seconds)
    return str(h + ':' + m + ':' + s)

def fillMissingIntervals_2(cell, intervalsALL, countHubs, tInterval):
    tInterval = tInterval*3600
    # cols of cell
    cell.reset_index(inplace=True)
    template = cell.loc[0,:]
    cell.set_index('Interval_start')
    # loop checks if all intervals exist
    for interval in intervalsALL:
        if interval not in intervalsALL:
            # if no value was interpolated for the interval
            template[-2]=interval
            cell.loc[interval, :] = template
            cell.loc[interval,'Interval_start'] = interval

    # sorting entries for one cell to be in correct temporal order
    cell.reset_index(inplace=True)
    cell = cell.sort_values(by=['Interval_start'], ascending=True, ignore_index=True)
    for z in range(0, countHubs):
        for direction in ['ac','eg']:
            if validateInterval(cell, z, direction):
                for i, row in cell.iterrows():
                    #travel and wait time are set to the duration of the interval
                    if str(row[str('Hub_' + str(z) + '_wait_time'+'_'+direction)]) == str(-999):
                        row[str('Hub_' + str(z) + '_wait_time'+'_'+direction)] = tInterval/2
                    if str(row[str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)])== str(-999):
                        if i+1 >= len(cell['Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction]):
                            row[str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)] = tInterval*0.8
                        elif str(cell.loc[i + 1, str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)]) == str(-999):
                            row[str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)] = tInterval*0.8
                        else:
                            row[str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)] = cell.loc[
                                i + 1, str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)]
                    cell.loc[i, :] = row
    return cell

def fillStopTimeTable(intervals, stop_times, cell, countHubs, d):
    if d == 'access':
        direction = 'ac'
    elif d =='egress':
        direction = 'eg'

    cell_id = cell.loc[0, 'cell_id']
    # sort by time to be able to iterate easily
    cell.sort_values(by=['Interval_start'], ascending=True, ignore_index=True, inplace=True)
    cell.set_index('Interval_start', inplace=True)
    #for all Hubs
    for z in range(0, countHubs, 1):
        if not validateInterval(cell, z, direction): continue
        i = 0
        tripNo = 1
        # setting start times
        currentInterval = intervals[i]
        currentEnd = intervals[i+1]
        overallEnd = intervals[-1]
        # setting departure parameters of current interval
        wt = int(cell.loc[currentInterval, str('Hub_' + str(z) + '_wait_time'+'_'+direction)])
        tt = int(cell.loc[currentInterval, str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)])
        if wt <= 0: wt = 30 # ensuring minumum headway of 60 seconds
        if tt <= 0: tt = 60 # ensuring mimimum travel time of 60 seconds
        headway = 2 * wt #wt is the average waiting time
        dep_stop1 = random.randint(intervals[i], intervals[i] + headway)
        arr_stop2 = dep_stop1 + tt

        while True:
            # Using current Interval settings until end of current Interval
            while dep_stop1 < currentEnd:

                j = len(stop_times['trip_id'])
                if direction == 'ac':
                    stop_times.loc[j, 'trip_id'] = str(str(cell_id) + '-Hub_' + str(z) + '>' + str(tripNo))
                    stop_times.loc[j, 'stop_id'] = cell_id
                elif direction == 'eg':
                    stop_times.loc[j, 'trip_id'] = str('Hub_' + str(z) +'-'+ str(cell_id) + '>' + str(tripNo))
                    stop_times.loc[j, 'stop_id'] = str('Hub_' + str(z))
                stop_times.loc[j, 'stop_sequence'] = 1
                stop_times.loc[j, 'departure_time'] = sec2str(dep_stop1)
                stop_times.loc[j, 'arrival_time'] = sec2str(dep_stop1)


                j = len(stop_times['trip_id'])
                if direction == 'ac':
                    stop_times.loc[j, 'trip_id'] = str(str(cell_id) + '-Hub_' + str(z) + '>' + str(tripNo))
                    stop_times.loc[j, 'stop_id'] = str('Hub_' + str(z))
                elif direction == 'eg':
                    stop_times.loc[j, 'trip_id'] = str('Hub_' + str(z) +'-'+ str(cell_id) + '>' + str(tripNo))
                    stop_times.loc[j, 'stop_id'] = cell_id
                stop_times.loc[j, 'departure_time'] = sec2str(arr_stop2)
                stop_times.loc[j, 'arrival_time'] = sec2str(arr_stop2)
                stop_times.loc[j, 'stop_sequence'] = 2

                dep_stop1 = dep_stop1 + headway
                arr_stop2 = dep_stop1 + tt

                tripNo = tripNo + 1

            i = i+1

            if currentEnd == overallEnd:
                return
                #return stop_times

            # updating the headway and waiting time with values of current interval
            currentInterval = intervals[i]
            currentEnd = intervals[i+1]
            # setting departure parameters of current interval
            wt = int(cell.loc[currentInterval, str('Hub_' + str(z) + '_wait_time'+'_'+direction)])
            tt = int(cell.loc[currentInterval, str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)])
            if wt <= 0: wt = 30  # ensuring minumum headway of 60 seconds
            if tt <= 0: tt = 60  # ensuring mimimum travel time of 60 seconds
            headway = 2 * wt #wt is the average waiting time
    return

def validateInterval(cell, z, direction):
    if str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction) not in cell.columns: return False
    tts = cell.loc[:, str('Hub_' + str(z) + '_<leg>_trav_time'+'_'+direction)].astype(str)
    ttS = tts.tolist()
    if ttS == ['-999'] * len(tts): return False
    elif ttS == [-999] * len(tts): return False
    else: return True

















