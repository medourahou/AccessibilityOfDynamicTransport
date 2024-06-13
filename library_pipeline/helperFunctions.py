import helperFunctions as hf
import pandas as pd
import logging
import tqdm

def Index2text(lineIndex,fileName,inputDim):
    if inputDim == 1:
        return lineIndex2text1D(lineIndex, fileName)
    elif inputDim == 2:
        return lineIndex2text2D(lineIndex, fileName)
    elif inputDim == 3:
        return lineIndex2text3D(lineIndex, fileName)
    else:
        return []

def lineIndex2text1D(lineIndex,fileName):
    with open(fileName, 'r') as fp:
        # To store lines
        lines = []
        for i, line in enumerate(fp):
            if i == lineIndex:
                txt = line.strip()
                fp.close()
                return txt
            elif i > lineIndex:
                break

def lineIndex2text2D(lineIs,fileName):
    with open(fileName, 'r') as fp:
        lines = []
        breakIndex = max(lineIs)
        for i, line in enumerate(fp):
            if i in lineIs:
                lines.append(line.strip())
            elif i > breakIndex:
                break
    fp.close()
    return lines

def lineIndex2text3D(lineContainer,fileName):
    f = open(fileName, 'r')
    fp = f.readlines()[0:-1]
    f.close()
    outputContainer = []
    progress = tqdm.tqdm(total=len(lineContainer), desc='Progress', position=0)
    for j, lineIs in enumerate(lineContainer):
        progress.update(1)
        lines = []
        breakIndex = max(lineIs)
        for i, line in enumerate(fp):
            if i in lineIs:
                sample = line.strip()
                lines.append(sample)
            elif i > breakIndex:
                i = 0
                break
        outputContainer.append(lines)
    return outputContainer

def createOutput(df,values):
    for i, val in enumerate(values):
        if len(val) == 1:
            values[i] = val[0]
    df.loc[len(df)] = values
    return df

def getPlansDimensions(plansFile, data):
    selected = False
    values = []
    filtervalueContainer = [i[:-1] for i in data]
    data.insert(0, 'agentID')
    filtervalueContainer.insert(0, 'agentID')
    [values.append([]) for i in data]
    plansDF = pd.DataFrame(columns=data)
    plans = open(plansFile, 'r')
    count = len(open(plansFile).readlines())

    progress = tqdm.tqdm(total=count, desc='Parsing lines for filter values', position=0)

    for lineIndex, line in enumerate(plans):
        progress.update(1)
        for filtervalue in filtervalueContainer:
            if line.find(filtervalue) != -1:
                if filtervalue == '<person':
                    values[0].append(line[line.find('id="') + 4:line.find('">')])
                    values[filtervalueContainer.index(filtervalue)].append(lineIndex)
                elif filtervalue == '<plan':
                    if line.find('selected="yes"') != -1:
                        values[filtervalueContainer.index(filtervalue)].append(lineIndex)
                        selected = True
                elif filtervalue == '</plan':
                    if selected:
                        values[filtervalueContainer.index(filtervalue)].append(lineIndex)
                        selected = False
                elif filtervalue == '</person':
                    values[filtervalueContainer.index(filtervalue)].append(lineIndex)
                    plansDF = hf.createOutput(plansDF, values)
                    values = []
                    [values.append([]) for i in data]
                else:
                    if selected:
                        values[filtervalueContainer.index(filtervalue)].append(lineIndex)
    plans.close()
    return plansDF

def addLegModes(df,plansFile):
    #better use addParameter function!
    dfCols = [[],[],[]]
    lineIndexConatainer = []
    for index, row in df.iterrows():
        lineIndexConatainer.append(row['<leg>'])
    legModes = []
    depT = []
    travT = []
    textContainer = Index2text(lineIndexConatainer, plansFile,3)
    for texts in textContainer:
        for txt in texts:
            legModes.append(txt[txt.find('mode="') + 6:txt.find('" dep_time')])
            depT.append(txt[txt.find('dep_time="') + 10:txt.find('" trav_time')])
            travT.append(txt[txt.find('trav_time="') + 11:txt.find('">')])
        dfCols[0].append(legModes)
        dfCols[1].append(depT)
        dfCols[2].append(travT)
        legModes = []
        depT = []
        travT = []
    df['legModes'] = dfCols[0]
    df['departureTimes'] = dfCols[1]
    df['travelTimes'] = dfCols[2]
    return df

def prepareKeys(keys,delimiter):
    startKeys = []
    if delimiter == '<>':
        for key in keys:
            startKeys.append(key + '="')
        startKeyLen = []
    elif delimiter == '{}':
        for key in keys:
            startKeys.append(key + '":')
        startKeyLen = []
    for key in startKeys:
        startKeyLen.append(len(key))
    return startKeys, startKeyLen

def extractSpecialAttribute(txt, startKey, startKeyLen, modeSpecificAttribute):
    a = txt.find('type="') + len('type="')
    b = txt.find('"', a)
    if txt[a:b] == modeSpecificAttribute:
        a = txt.find(startKey) + startKeyLen
        if txt[a] == '"':
            a = a+1
        if txt.find(',"') != -1:
            b = txt.find(',"', a)
        else:
            b = txt.find('}', a)
        return txt[a:b]
    else:
        return ''

def addAttribute(df,plansFile,keys,dfXmlLineKey,optional):
    if len(optional) > 1:
        delimiter = optional[0]
        modeSpecificAttribute = optional[1]
    else:
        delimiter = "<>"
    startKeys, startKeyLen = prepareKeys(keys, delimiter)
    lineIndexConatainer = []
    for index, row in df.iterrows():
        lineIndexConatainer.append(row[dfXmlLineKey])
    values = []
    dfCols = []
    for i in range(len(startKeys)):
        values.append([])
        dfCols.append([])
    textContainer = Index2text(lineIndexConatainer, plansFile,3)
    for texts in textContainer:
        for txt in texts:
            for i, key in enumerate(startKeys):
                if delimiter == '<>':
                    a = txt.find(startKeys[i]) + startKeyLen[i]
                    b = txt.find('"',a)
                    values[i].append(txt[a:b])
                elif delimiter == '{}':
                    value = extractSpecialAttribute(txt, startKeys[i], startKeyLen[i], modeSpecificAttribute)
                    values[i].append(value)
                else:
                    logging.error('No valid delimiter given. Choose from <> and {}')

        for i, val in enumerate(values):
            dfCols[i].append(val)
        values = []
        for i in range(len(startKeys)):
            values.append([])
    for i, col in enumerate(dfCols):
        df[dfXmlLineKey + '_' + keys[i]] = col
    return df

def filterModePlans(df,mode,cols):
    dfFiltered = df.copy()
    dfFiltered.drop(dfFiltered.index, inplace=True)
    for index, row in df.iterrows():
        if mode in row['<leg>_mode']:
            dfFiltered.loc[len(dfFiltered)] = row
    return dfFiltered









