import pandas as pd
from zipfile import ZipFile
import os

import logging

# the readGTFS, joinGTFS, exportGTFS function are used to read in the relevant .txt files from a gtfs zip-archive
# and put them into pandas (pd) dataframes. The columns of the dataframes are named equally to the gtfs standard.
# Further information can be found here: https://developers.google.com/transit/gtfs/
# The various dataframes are stored in a dictionary. Each entry of the dict can be accesses with the known gtfs filename
# Example:
# agency.txt --> gtfs['agency']

# run(path, gtfs) runs the read, join, export functions in the right order
# combined are the newly imported gtfs from read() and the given gtfs (drt gtfs)
# the finished gtfs zip archive is output to the given path where also the original gtfs was found
def run(path,gtfs):
    defaultGTFS = readGTFS(path)
    gtfs = joinGTFS(defaultGTFS, gtfs)
    exportGTFS(path, gtfs)

# opening the .zip archive at location gtfsPath, reading the files directly into pandas dataframes and storing them
# in a dictionary. The dict is returned
def readGTFS(gtfsPath):
    with ZipFile(gtfsPath + 'gtfs.zip', 'r') as f:
        gtfs = {
            "agency": pd.read_csv(f.open('agency.txt'), sep=',', header=0),
            "calendar_dates": pd.read_csv(f.open('calendar_dates.txt'), sep=',', header=0),
            "calendar": pd.read_csv(f.open('calendar.txt'), sep=',', header=0),
            "routes": pd.read_csv(f.open('routes.txt'), sep=',', header=0),
            "stop_times": pd.read_csv(f.open('stop_times.txt'), sep=',', header=0, low_memory=False),
            "stops": pd.read_csv(f.open('stops.txt'), sep=',', header=0),
            "trips": pd.read_csv(f.open('trips.txt'), sep=',', header=0)
        }
    return gtfs

# Taking two dictionaries of gtfs as input. The entries of the dict must be the dataframes of a gtfs as described above.
# The pandas dataframes are merged using the pd.concat() function. A new index is given and all cells that are not
# filled are set to NA
# The resulting new dict is returned
def joinGTFS(gtfs, drt_gtfs):
    gtfsJOINED = {
        "agency": pd.concat([gtfs['agency'],drt_gtfs['agency']], ignore_index=True,sort=False),
        "calendar_dates": pd.concat([gtfs['calendar_dates'],drt_gtfs['calendar_dates']], ignore_index=True,sort=False),
        "calendar": pd.concat([gtfs['calendar'],drt_gtfs['calendar']], ignore_index=True,sort=False),
        "routes": pd.concat([gtfs['routes'],drt_gtfs['routes']], ignore_index=True,sort=False),
        "stop_times": pd.concat([gtfs['stop_times'],drt_gtfs['stop_times']], ignore_index=True,sort=False),
        "stops": pd.concat([gtfs['stops'],drt_gtfs['stops']], ignore_index=True,sort=False),
        "trips": pd.concat([gtfs['trips'],drt_gtfs['trips']], ignore_index=True,sort=False)
    }
    return gtfsJOINED


# writing the pandas dataframes into txt files and exporting them.
# the files are added to a gtfs zip archive and removed from the disk.
def exportGTFS(outputPath, gtfs):
    logging.info('exporting combined GTFS zip archive')
    filenames = ['stop_times','stops','trips','calendar','calendar_dates','routes','agency']
    # creating zip archive
    zipObj = ZipFile(outputPath + '/gtfsOUT.zip', 'w')
    for f in filenames:
        # writing .txt file temporarily
        gtfs[f].to_csv(outputPath + f + ".txt", sep=',', index=False)
        # adding all gtfs text files to the zip archive
        zipObj.write(outputPath + f +".txt", f + ".txt")
        # removing temporary .txt file
        os.remove(outputPath + f + ".txt")
    # close the Zip File
    zipObj.close()