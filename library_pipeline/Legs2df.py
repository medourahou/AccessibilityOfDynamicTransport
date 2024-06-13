import logging
import pandas as pd
from library_pipeline import ExtractDrtTrips as edt
import logging

def run(path):
    logging.info('Importing Trips from output_legs.csv file (Matsim Output)')
    df = pd.read_csv(path, sep=';', header=0)
    df = df[df["mode"]=='drt']
    df.rename(columns = { "person" : "agentID",
                           "dep_time" : '<leg>_dep_time',
                            "trav_time" : '<leg>_trav_time',
                            "start_x" : 'start_x',
                            "start_y" : 'start_y',
                            "end_x" : 'end_x',
                            "end_y" : 'end_y',
                            "distance" : '<route>_distance'
                          },inplace=True)
    df.drop(columns=['access_stop_id',
                     'egress_stop_id',
                     'transit_line',
                     'transit_route',
                     'mode'],
                    inplace=True)
    df = time2seconds(df)
    return df

def time2seconds(df):
    travTime=[]
    waitTime=[]
    depTime=[]
    for i, row in df.iterrows():
        waitTime.append(edt.convertValues(row['wait_time']))
        travTime.append(edt.convertValues(row['<leg>_trav_time']) - waitTime[-1])
        depTime.append(edt.convertValues(row['<leg>_dep_time']))
    df.drop(columns=["<leg>_trav_time","wait_time",'<leg>_dep_time'], inplace=True)
    df["<leg>_trav_time"] = travTime
    df["wait_time"] = waitTime
    df['<leg>_dep_time'] = depTime
    return df




