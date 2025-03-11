import logging
import numpy as np
import pandas as pd

def run(df,cutoff=100):
    cutoff = cutoff
    logging.info('Extracting Hubs based on trip data')
    dforigin = df[['agentID', 'origin']]
    dforigin = dforigin.groupby(['origin'])['agentID'].count().reset_index(name="Count")
    dforigin = dforigin[dforigin['Count'] > cutoff]
    dforigin.rename(columns={'origin': 'position'}, inplace=True)

    dfdest = df[['agentID', 'destination']]
    dfdest = dfdest.groupby(['destination'])['agentID'].count().reset_index(name="Count")
    dfdest = dfdest[dfdest['Count'] > cutoff]
    dfdest.rename(columns={'destination': 'position'}, inplace=True)

    dfhubs = pd.concat([dforigin, dfdest]).drop_duplicates(subset=['position']).reset_index().drop(columns=['index'])
    hubIDs = []
    for i in range(len(dfhubs)):
        hubIDs.append('Hub_' + str(i))
    dfhubs['name'] = hubIDs
    #dfhubs.to_csv("hubs.csv", sep=';')
    return dfhubs

def assignHub2Trip(dfHubs, dfTrips):

    dfTrips['tripType'] = np.where(dfTrips['origin'].isin(dfHubs['position']), "egress", "access")
    hubs = []
    for i, hub in dfHubs.iterrows():
        conditions = [
            (dfTrips['origin'] == hub['position']) & (dfTrips['tripType'] == "egress"),
            (dfTrips['destination'] == hub['position']) & (dfTrips['tripType'] == "access")
            ]
        tierValues = [hub['name'],hub['name']]
        hubs.append(np.select(conditions, tierValues).tolist())
        # dfTrips['Hub'] = np.select(conditions, tierValues)

    nHubs = len(hubs)
    Hub = []
    for i in range(len(hubs[0])):
        for j in range(nHubs):
            if hubs[j][i] != '0':
                Hub.append(hubs[j][i])
                break
    dfTrips['Hub'] = Hub



