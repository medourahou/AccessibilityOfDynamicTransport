import os
import zipfile
import pandas as pd
import pymongo as pym
import time
from datetime import datetime
import numpy as np
infty = 1000000

def printGtfsDate(directoryGTFS):
    print("interval of validity of the gtfs files")
    for filename in os.listdir(directoryGTFS):
        if filename.endswith(".zip"):

            archive = zipfile.ZipFile(directoryGTFS + filename, 'r')
            for fileTxt in archive.filelist:
                if fileTxt.filename =="calendar.txt":
                    lines = archive.open("calendar.txt", mode='r')
                    res = pd.read_csv(lines,encoding = 'utf-8-sig', dtype=str)

                    timePrint = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(os.path.getmtime(directoryGTFS + filename)))
                    print("{0} file\n calendar.txt -> start_date:{1}, end_date:{2} (first row)".format(filename, res['start_date'][0],res['end_date'][0] ))
                if fileTxt.filename =="calendar_dates.txt":
                    lines = archive.open("calendar_dates.txt", mode='r')
                    res = pd.read_csv(lines,encoding = 'utf-8-sig', dtype=str)

                    timePrint = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(os.path.getmtime(directoryGTFS + filename)))
                    print("{0} file\n calendar_dates.txt -> date:{1} (first row)".format(filename, res['date'][0]))

def file_len(fname):
    with archive.open(nameCSVFile, mode='rU') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def readConnections(gtfsDB, city, directoryGTFS, day, dayName, overwrite=False):
    """Read and store connections in MongoDB with day information"""
    # If overwrite is True, delete existing connections
    if overwrite:
        gtfsDB['connections'].delete_many({'city': city})
    
    services_id = readValidityDateGtfs(gtfsDB, day, dayName, city)

    tripID2Route = {}
    count = 0
    for trip in gtfsDB['trips'].find({'city': city}):
        try:
            tripID2Route[trip['file']][str(trip['trip_id'])] = str(trip['route_id'])
        except:
            tripID2Route[trip['file']] = {str(trip['trip_id']): str(trip['route_id'])}
        if count % 1000 == 0:
            print('number of trips {0}'.format(count), end="\r")
        count += 1

    stopsTimes = {}
    for filename in reversed(list(os.listdir(directoryGTFS))):
        if filename.endswith(".zip"):
            stopsTimes = {}
            archive = zipfile.ZipFile(directoryGTFS + filename, 'r')
            print('\n', filename)
            
            # Process stop_times.txt
            collGtfs = gtfsDB['stop_times'].find()
            if 'stop_times.txt' in archive.namelist():
                lines = archive.open('stop_times.txt', mode='r')
                res = pd.read_csv(lines, encoding='utf-8-sig', dtype=str)
                res["city"] = city
                res["file"] = filename
                res["day"] = day  # Add day information
                res = list(v for v in res.to_dict("index").values())
                res = list(collGtfs)
                
                tot = len(res)
                for i, elem in enumerate(res):
                    print('{0}, {1}'.format(count, tot), end="\r")
                    count += 1
                    elem['posStop'] = str(elem['stop_id'])
                    elem['route'] = tripID2Route[filename][str(elem['trip_id'])]
                    try:
                        stopsTimes[elem['trip_id']].append(elem)
                    except:
                        stopsTimes[elem['trip_id']] = [elem]
                
                fillConnections(gtfsDB, stopsTimes, services_id[filename], city, filename, archive, day)
    
    indexConnections(gtfsDB)

def indexConnections(gtfsDB):
    gtfsDB['connections'].create_index([("pStart", pym.ASCENDING),("city", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("pEnd", pym.ASCENDING),("city", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("tStart", pym.ASCENDING),("tEnd", pym.ASCENDING),("city", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("tEnd", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("city", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("trip_id", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("route_id", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("file", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("city", pym.ASCENDING)])
    gtfsDB['connections'].create_index([("day", pym.ASCENDING)])

def checkNumberOfGtfs(gtfsDB, city):
    namefiles={}
    for name in gtfsDB['calendar'].distinct('file', filter={'city': city} ):
        try:
            namefiles[name] += 1
        except:
            namefiles[name] = 1
    for name in gtfsDB['calendar_dates'].distinct('file', filter={'city': city} ):
        try:
            namefiles[name] += 1
        except:
            namefiles[name] = 1
    #print namefiles

    print ('number of file in calendar+calendar_dates: {0}\nin stops: {1}'.format(len(namefiles), len(gtfsDB['stops'].distinct('file', filter={'city': city}))))
    return namefiles

def readValidityDateGtfs(gtfsDB, day, dayName, city):
    namefiles = checkNumberOfGtfs(gtfsDB, city)
    services_id = {}
    print("\nChecking the number of services active in the date selected:")
    for serv in gtfsDB['calendar'].find({dayName : '1','city':city}):
        #print serv['end_date'], serv
        try:
            services_id[serv['file']].append(serv['service_id'])
        except:
            services_id[serv['file']] = [serv['service_id']]

    for name in namefiles:
        try:
            services_id[name]
            print( 'file: {0} \t total number of active service (in calendar.txt): {1}'.format(name, len(services_id[name])))
        except KeyError:
            print( 'file: {0} \t total number of active service (in calendar.txt): {1}'.format(name, 'Serv NOT FOUND!!'))


    print( 'number of different service_id:', len(services_id))
    print ('\n')

    for exp in  gtfsDB['calendar_dates'].find({'date':day,'city':city}):
        if(exp['exception_type'] == '1' or exp['exception_type'] == 1):
            try:
                services_id[exp['file']].append(exp['service_id'])
            except:
                services_id[exp['file']] = [exp['service_id']]
        else:
            if exp['service_id'] in services_id[exp['file']] : services_id[exp['file']].remove(exp['service_id'])
    tot = 0
    for name in  namefiles:
        try:
            tot += len(services_id[name])
            print ('file: {0} \t total number of active service (in calendar_dates.txt): {1}'.format(name, len(services_id[name])))
        except KeyError:
            print( 'file: {0} \t total number of active service (in calendar_dates.txt): {1}'.format(name, 'Serv NOT FOUND!!'))
    print( 'number of different service_id:', len(services_id), 'total number of active services found:', tot)
    return services_id


def findSec(hour):
    hour = str(hour)
    if(len(hour)>3):
        if(len(hour) == 8):
            if(int(hour[0:2]) >= 24):
                hourInt = int(hour[0:2]);
                diffInt = int(hour[0:2]) - 24;
                timeToCompute = str(diffInt) + hour[2:]
                #print timeToCompute;
                pic = datetime.strptime(timeToCompute, "%H:%M:%S")
                #print timeToCompute, hourInt*3600 + pic.hour*3600 + pic.minute*60 + pic.second
                return hourInt*3600 + pic.hour*3600 + pic.minute*60 + pic.second
            else:
                if(hour[0] == ' '): hour = hour[1:]
                pic = datetime.strptime(hour, "%H:%M:%S")
                return pic.hour*3600 + pic.minute*60 + pic.second
        else:
            #print len(hour)
            pic = datetime.strptime(hour, "%H:%M:%S")
            return pic.hour*3600 + pic.minute*60 + pic.second
    else:
        return infty;

def fillConnections(gtfsDB, stopsTimes, services_id, city, filename, archive, day):
    count = 0
    count_err = 0
    count_err_start = 0
    count_err_start_after = 0
    tot = gtfsDB['trips'].count_documents({'service_id': {'$in': services_id}, 'city': city, 'file': filename})

    listTrip = list(gtfsDB['trips'].find({'service_id': {'$in': services_id}, 'city': city, 'file': filename}))
    listToInsert = []
    listOfFreqTrip = {}

    # Handle frequencies.txt if it exists
    if 'frequencies.txt' in archive.namelist():
        lines = archive.open('frequencies.txt', mode='r')
        res = pd.read_csv(lines, encoding='utf-8-sig', dtype=str)
        res["city"] = city
        res["file"] = filename
        res = list(v for v in res.to_dict("index").values())
        for freq in res:
            try:
                listOfFreqTrip[freq['trip_id']].append(freq)
            except KeyError:
                listOfFreqTrip[freq['trip_id']] = [freq]
        if len(res) > 0:
            print("found freq for # of trips", len(res))

    for trip in listTrip:
        if(trip['trip_id'] in stopsTimes):
            resNotSorted = stopsTimes[trip['trip_id']]
            res = sorted(resNotSorted, key=lambda k: int(k['stop_sequence']))
            
            # Handle frequency-based trips
            if trip['trip_id'] in listOfFreqTrip:
                for freq in listOfFreqTrip[trip['trip_id']]:
                    startTime = findSec(freq['start_time'])
                    endTime = findSec(freq['end_time'])
                    currentTime = startTime
                    startTrip = startTime
                    count = 0
                    if len(res) > 0:
                        while True:
                            for i, stop in enumerate(res[:-1]):
                                diff = findSec(res[i+1]['arrival_time']) - findSec(res[i]['departure_time'])

                                objToInsert = {
                                    'pStart': res[i]['posStop'],
                                    'pEnd': res[i+1]['posStop'],
                                    'tStart': currentTime,
                                    'tEnd': currentTime + diff,
                                    'trip_id': res[i]['trip_id'],
                                    'route_id': res[i]['route'],
                                    'seq': res[i]['stop_sequence'],
                                    'file': res[i]['file'],
                                    'city': city,
                                    'day': day  # Add day information
                                }
                                if(objToInsert['tStart'] > objToInsert['tEnd']):
                                    count_err += 1
                                else:
                                    listToInsert.append(objToInsert)
                                currentTime += diff
                            startTrip += int(freq['headway_secs'])
                            currentTime = startTrip
                            if(startTrip > endTime):
                                break
                            count += 1

            # Handle regular trips
            else:
                if len(res) > 1:
                    startStop = res[0]
                    endStop = res[len(res)-1]
                    startTime = findSec(startStop['departure_time'])
                    endTime = findSec(endStop['arrival_time'])
                    
                    if(startTime == infty or endTime == infty):
                        count_err_start += 1
                    else:
                        for i, stop in enumerate(res[:-1]):
                            res[i]['departure_time'] = res[i-1]['departure_time'] if findSec(res[i]['departure_time']) == infty else res[i]['departure_time']
                            res[i]['arrival_time'] = res[i-1]['arrival_time'] if findSec(res[i]['arrival_time']) == infty else res[i]['arrival_time']
                            tEnd = findSec(res[i]['arrival_time']) if findSec(res[i+1]['arrival_time']) == infty else findSec(res[i+1]['arrival_time'])
                            
                            if(findSec(res[i+1]['arrival_time']) == infty):
                                count_err_start_after += 1

                            objToInsert = {
                                'pStart': res[i]['posStop'],
                                'pEnd': res[i+1]['posStop'],
                                'tStart': findSec(res[i]['departure_time']),
                                'tEnd': tEnd,
                                'trip_id': res[i]['trip_id'],
                                'route_id': res[i]['route'],
                                'seq': res[i]['stop_sequence'],
                                'file': res[i]['file'],
                                'city': city,
                                'day': day  # Add day information
                            }
                            if(findSec(res[i]['departure_time']) > tEnd):
                                count_err += 1
                            else:
                                listToInsert.append(objToInsert)

        print('count {0}, tot {1}, err {2}, err_start {3}, err_start_after {4}'.format(
            count, tot, count_err, count_err_start, count_err_start_after), end="\r")
        count += 1

    if(len(listToInsert) > 0):
        print('inserting to DB....')
        gtfsDB['connections'].insert_many(listToInsert)
    print('tot connections', len(listToInsert))


def updateConnectionsStopName(gtfsDB, city):
    # tot = gtfsDB['stops'].find({'city':city}).count()
    tot = gtfsDB['stops'].count_documents({'city':city})
    count = 0
    totC = gtfsDB['connections'].count_documents({'city':city})
    # totC = gtfsDB['connections'].find({'city':city}).count()
    c1 = 0
    c2 = 0
    stops =list(gtfsDB['stops'].find({'city':city}))
    length=len(stops)
    for stop in range(0,length):
        #print("stop number ", count)
        res1 = gtfsDB['connections'].update_many({'city':city,'pStart':stops[stop]['stop_id'],'file':stops[stop]['file']},{"$set":{'pStart':stops[stop]['pos'],"updatedStart":True}})
        #print("pStart done...")
        res2 = gtfsDB['connections'].update_many({'city':city,'pEnd':stops[stop]['stop_id'],'file':stops[stop]['file']},{"$set":{'pEnd':stops[stop]['pos'],"updatedEnd":True}})
        #print("pEnd done...")
        c1 += res1.modified_count
        c2 += res2.modified_count
        print ('\r{0},{1}-- pStart {2} pEnd {3}, totC {4}'.format(count,tot,c1,c2, totC),end="\r")
        count += 1
    delStart = gtfsDB['connections'].delete_many({'city':city, 'updatedStart':{"$exists":False}}).deleted_count
    delEnd = gtfsDB['connections'].delete_many({'city':city, 'updatedEnd':{"$exists":False}}).deleted_count
    print("connections deleted",delEnd + delStart)

def makeArrayConnectionsOld(gtfsDB, hStart, city):
    fields = {'tStart':1,'tEnd':1, 'pStart':1, 'pEnd':1, '_id':0 }
    pipeline = [
        {'$match':{'city': city,'tStart':{'$gte':hStart}}},
        {'$sort':{'tStart':1}},
        {'$project':{'_id':"$_id", "c":['$tStart', '$tEnd','$pStart','$pEnd']}},
    ]
    allCC = gtfsDB['connections'].aggregate(pipeline)
    arrayCC = np.full((gtfsDB['connections'].find({"city":city,'tStart':{'$gte':hStart}}).count(),4),1.,dtype = np.int)
    countC = 0
    tot = gtfsDB['connections'].find({'tStart':{'$gte':hStart},'city':city}).count()
    for cc in allCC:
        #if round(cc['tStart']) <=  round(cc['tEnd']) and isinstance(cc['pStart'] , int ) and isinstance(cc['pEnd'] , int ):
        print(' {0}, {1}, {2}'.format(countC, tot, cc['c']),end="\r");
        #try:
        cc['c'] = [round(int(c)) for c in cc['c']]
        arrayCC[countC] = cc['c']
        countC += 1
        #except:
            #print('error')
    print( 'Num of connection', len(arrayCC))
    return arrayCC
#new 
def makeArrayConnections(gtfsDB, hStart, city, day=None):
    """Make array of connections for specific day"""
    print("start making connections array")
    
    # Build query including day if specified
    typeMatch = {
        'city': city,
        'tStart': {'$gte': hStart, "$type": "number"},
        'tEnd': {"$type": "number"},
        'pStart': {"$type": "number"},
        'pEnd': {"$type": "number"}
    }
    
    if day is not None:
        typeMatch['day'] = day
    
    pipeline = [
        {'$match': typeMatch},
        {'$sort': {'tStart': 1}},
        {'$project': {'_id': "$_id", "c": ['$tStart', '$tEnd', '$pStart', '$pEnd']}}
    ]
    
    allCC = list(gtfsDB['connections'].aggregate(pipeline))
    print("done recover all cc", len(allCC))
    allCC = np.array([x["c"] for x in allCC])
    print("converted")
    print('Num of connection', len(allCC))
    return allCC


