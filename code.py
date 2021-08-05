# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:11:31 2021

@author: Aditya Chandran
"""

import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import os
import numpy as np
import shapefile as shp
import sklearn
from dask import dataframe as ddf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import median
from statistics import mean
from scipy.interpolate import make_interp_spline
pd.options.mode.chained_assignment = None



fig1 = plt.figure()
ax = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)
#fig1, ax = plt.subplots(nrows=1, ncols=1)
#ax2 = fig1.add_subplot(2, 1, 2)
# fig2 = plt.figure()
# fig3 = plt.figure()
# fig4 = plt.figure()
# fig5 = plt.figure()
# fig6 = plt.figure()
# fig7 = plt.figure()
# fig8 = plt.figure()


NUM_OF_FIGS = 8

figlist = []  # [fig1]#, fig2]  # , fig3, fig4, fig5, fig6, fig7, fig8]


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#reads in a dataframe from a .dat file without reading metadata, 
#delimiter is whitespace, and adds PRN as a column
def readFile(name):
    df = ddf.read_csv(name, delim_whitespace=True, skiprows=5, names=["number", "day", "hour", "minute",
                                                                     "second", "TEC", "local time",
                                                                     "elevation of looking direction",
                                                                     "latitude",
                                                                     "longitude", "slant TEC", "azimuth", "PRN"])
    PRN = 0
    for i in range(len(df.index)):
        if df.iloc[i, 0] == "PRN":
            PRN = df.iloc[i, 1]
        else:
            df.at[i, "PRN"] = PRN
    return df

def fastReadFile(name, rows):
    df = ddf.read_csv(name, delim_whitespace=True, skiprows=rows, names=["number", "day", "hour", "minute",
                                                                         "second", "TEC", "local time",
                                                                         "elevation of looking direction",
                                                                         "latitude",
                                                                         "longitude", "slant TEC", "azimuth"])
    return df



#reads in the full file from a .dat or csv file, delimiter is whitespace
def readFullFile(name):
    df = pd.read_csv(name, delim_whitespace=True, names=["number", "day", "hour", "minute",
                                                         "second", "TEC", "local time",
                                                         "elevation of looking direction",
                                                         "latitude",
                                                         "longitude", "slant TEC", "azimuth", "PRN"])
    return df

#filters and returns a dataframe containing only rows from the 
#input dataframe that have an hour value between hour and hour+interval
def filterFile(df, hour, interval):
    filteredDf = df.loc[(df["hour"] >= hour) & (df["hour"] < hour+interval)]
    return filteredDf


#splits a parent dataframe into a list of dataframes ordered by PRN
def splitParentDf(filename):
    dflist = []
    parentDf = readFile(filename)
    for i in range(33):
        if i != 0:
            dflist.append(parentDf.loc[parentDf["PRN"] == i])
    return dflist


#uses the filterFile() function to iterate through and filter a 
#list of dataframes
def filterDfList(dflist, hour):
    filteredDfList = []
    for i in range(len(dflist)):
        filterdf = filterFile(dflist[i], hour, 2)
        if not filterdf.empty:
            filteredDfList.append(filterdf)
    return filteredDfList


#graphs a list of dataframes onto multiple subplots, color is 
#based on TEC value
def graphDflist(dflist, numGraphs):
    figCount = 0
    flag = 0
    for i in range(numGraphs):
        if flag > 3:
            flag = 0
            figCount += 1
        ax1 = figlist[figCount].add_subplot(2, 2, flag + 1)
        ax1.scatter(dflist[i]["longitude"], dflist[i]["latitude"], c=dflist[i]["TEC"], cmap=plt.cm.coolwarm)
        flag += 1
    for x in range(NUM_OF_FIGS - figCount):
        plt.close(figlist[x + figCount + 1])
    plt.show()


#plots the locations of the LISN stations
def graphStation(XYlist):
    ax.scatter(XYlist["x"], XYlist["y"], s=20, marker="^", facecolors="none", edgecolors='k', label="station")


#graphs TEC data from one dataframe, color is based on TEC value
def graphDf(df, x, y, c):
    sc1 = ax.hexbin(df[x], df[y], C=df[c], vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    divider = make_axes_locatable(ax)
    caxes = divider.append_axes('right', size='2%', pad=0.1)

    fig1.colorbar(sc1, cax=caxes, orientation='vertical', ticks=[0, 70])
    
    
    
def graphDfNoColorbar(df, x, y, c):
    sc1 = ax.hexbin(df[x], df[y], C=df[c], vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    divider = make_axes_locatable(ax)
    caxes = divider.append_axes('right', size='2%', pad=0.1)

    #fig1.colorbar(sc1, cax=caxes, orientation='vertical', ticks=[0, 70])
    caxes.set_visible(False)
    caxes.get_xaxis().set_ticks([])
    caxes.get_yaxis().set_ticks([])
    caxes.get_xaxis().set_ticklabels([])
    caxes.get_yaxis().set_ticklabels([])
    return sc1, caxes
    
def clippedcolorbar(axis, CS, **kwargs):
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    fig = fig1
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    kwargs['cax'] = axis
    return fig.colorbar(m, **kwargs)
    
def scatterDf(df, x, y, c):
    Z = df.pivot_table(index=x, columns=y, values=c).T.values

    X_unique = np.sort(df.day.unique())
    Y_unique = np.sort(df.lat.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    #levels = np.linspace(0.0, 70.0, 2)
    sc1 = ax2.contourf(X, Y, Z, 100, vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    divider = make_axes_locatable(ax2)
    caxes = divider.append_axes('right', size='2%', pad=0.1)

    clippedcolorbar(caxes, sc1)

def scatterDfNoColorbar(df, x, y, c):
    Z = df.pivot_table(index=x, columns=y, values=c).T.values

    X_unique = np.sort(df.day.unique())
    Y_unique = np.sort(df.lat.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    #levels = np.linspace(0.0, 70.0, 2)
    sc1 = ax.contourf(X, Y, Z, 100, vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    #divider = make_axes_locatable(ax)
    #caxes = divider.append_axes('right', size='2%', pad=0.1)

    #clippedcolorbar(caxes, sc1)

#iterates through a list of filtered dataframes and calculates 
#the hour for which the most data points will be available for graphing
def calcOptimalHour(dflist):
    listOfFills = []
    for i in range(24):
        numOfFilledDfs = 0
        for x in range(32):
            calclist = filterFile(dflist[x], i, 2)
            if not calclist.empty:
                numOfFilledDfs += 1
        listOfFills.append(numOfFilledDfs)
    return [listOfFills.index(max(listOfFills)), max(listOfFills)]


#returns a dataframe containing only every n'th element of the parent
def decimateDf(df, freq):
    df2 = pd.DataFrame(columns=['lon', 'lat', 'alt', 'tec', 'NmF2', 'HmF2'])
    for i in range(len(df)):
        if i % freq == 0:
            df2.loc[i] = df.iloc[i]
            print(i)
    return df2


#reads in a NETCDF4 file using the netCDF4 library
def read(filename):
    nc = Dataset(filename, format="NETCDF4")
    lon = nc.variables["lon"]
    lat = nc.variables["lat"]
    tec = nc.variables['tec']

    df = pd.DataFrame(columns=['lon', 'lat', 'tec'])
    counter = 0
    for i in range(len(lat)):
        for j in range(len(lon)):
            df.loc[counter] = [lon[j], lat[i], tec[i, j]]
            counter += 1
    print("done.", counter, "elements processed.")
    return df


#reads through a directory and returns every file with a 
#matching day value
def readDirectoy(pathToDir, strDay):
    filelist = []
    directory = pathToDir
    try:
        for entry in os.scandir(directory):
            if entry.path.endswith(".dat") and entry.is_file():
                filelist.append(entry.path)
    except FileNotFoundError:
        exit("Invalid path to directory")
    filteredFileList = []
    for i in filelist:
        if strDay in i:
            filteredFileList.append(i)
    if len(filteredFileList) == 0:
        exit("No files found matching these parameters")
    else:
        return filteredFileList


#converts a path or a list of paths to filtered dataframes 
#using the filterFile() function
def convertPaths(paths, hour, interval):
    files = []
    fullFiles = []
    for i in paths:
        files.append(readFile(i))
        fullFiles.append(readFullFile(i))
    filteredFiles = []
    for i in files:
        filteredFiles.append(filterFile(i, hour, interval))
    return filteredFiles, fullFiles


#converts a list of filtered dataframes into csv files 
#containing only latitude, longitude, and TEC values
def processFiles(files):
    filelist = []
    for i in files:
        filelist.append(i[["TEC", "latitude", "longitude", "hour"]])
    processedDf = pd.concat(filelist)
    return processedDf


#creates a file, just to clean up things because dataframe.to_csv looks bad
def createFile(dataframe, filename):
    dataframe.to_csv(filename)


#creates a file containing the longitude and latitude values of every station
#used to gather data
def stationLocations(fullFileList):
    stationXYs = pd.DataFrame(columns=["x", "y"])
    counter = 0
    for i in fullFileList:
        stationXYs.loc[counter] = ([float(i.iloc[1, 1]), float(i.iloc[1, 0])])
        counter += 1
    return stationXYs


#Graphs a the given indicies of a shapefile
def graphOutline(path, indexArr):
    shapeIndicies = indexArr
    sf = shp.Reader(path)
    for x in shapeIndicies:
        shape = sf.shape(x)
        points = np.array(shape.points)
        intervals = list(shape.parts) + [len(shape.points)]
        for (i, j) in zip(intervals[:-1], intervals[1:]):
            plt.plot(*zip(*points[i:j]), "-", color="black")
            

#retrieves TEC plot data spanning one month, with each day 
#corresponding to a consistent line of longitude
def makeMonthPlotData(lonValue, path, pathToFile, dateRange):
    files = []
    filenames = []
    counter = 0
    
    for entry in os.scandir(path):
        if entry.path.endswith(".dat") and entry.is_file():
            files.append(entry.path)
            filenames.append(entry.name)
    print("directory scanned")
    for item in files:
        day = int(filenames[counter][-7:-4])
        if day >= dateRange[0] and day <= dateRange[1]:
            j = fastReadFile(item, 5)
            z = (j.loc[(j["longitude"] > lonValue-1) & (j["longitude"] < lonValue+1) & (j["elevation of looking direction"] >= 30)])
            z["day"] -= 5844
            cut = z[["day", "latitude", "TEC", "hour"]]
            if len(cut.index) != 0:
                cut.to_csv(pathToFile, header=False, mode='a')
        counter += 1
    print("file created")
    
    
#retrieves TEC plot data spanning one year, with each day 
#corresponding to a consistent line of longitude from the model data
def ncMonthPlotData(lonValue, path, pathToFile):
    files = []
    filenames = []
    counter = 0
    
    for entry in os.scandir(path):
        if entry.path.endswith(".nc") and entry.is_file():
            files.append(entry.path)
            filenames.append(entry.name)
    print("directory scanned")
    for item in files:
        date = int(filenames[counter][8:12])
        day = int(filenames[counter][9:12])
        hour = int(filenames[counter][13:15])
        year = int(filenames[counter][4:8])
        cut = pd.DataFrame(columns = [["lon", "lat", "tec", "day"]])
        if date >= 101 and date <= 131 and hour == 20 and year == 2016:
            j = read(item)
            z = (j.loc[(j["lon"] > lonValue-1) & (j["lon"] < lonValue+1)])
            cut = z[["lon", "lat", "tec"]]
            cut["day"] = day
            if len(cut.index) != 0:
                cut.to_csv(pathToFile, header=False, mode='a')
        counter += 1
    print("file created")

#averages all the data within specified 2-degree bins, like -90 to -88 degrees latitude
def averageInDegree(df):
    binlist = []
    counter = 0
    returndf = pd.DataFrame(columns = ["day", "latitude", "TEC", "hour", "elevation of looking direction"])
    counts = pd.DataFrame(columns = ["latitude", "count", "day"])
    minVal = min(df["latitude"])
    binlist.append(minVal)
    maxVal = max(df["latitude"])
    i = minVal
    print(minVal, maxVal)
    while i <= maxVal:
        i+=2
        binlist.append(i)
    averager = df.groupby(pd.cut(df["latitude"], bins=binlist))
    keys = averager.groups.keys()
    print("done binning")
    for z in keys:
        try:
            interval = averager.get_group(z)
            grids = interval.groupby('day')
            gridKeys = grids.groups.keys()
        except:
            print("continuing")
            continue
        for j in gridKeys:
            box = grids.get_group(j)
            box["TEC"] = box["TEC"].mean()
            counts.loc[counter] = [box["latitude"].median(), len(box["TEC"]), j]
            returndf = returndf.append(box)
            counter+=1
    return returndf, counts

#Loops through a directory and makes one file for each month of LISN data.
#Filters input files so that the longitude matches lonValue, and so that the
#angle of elevation is greater than or equal to 30 degrees
def makeYearPlotData(lonValue):
    counter = 0
    previous = 0
    
    dateRanges = [29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    for i in months:
        print(i)
        monthData = [previous + 1, previous + dateRanges[counter]]
        previous = monthData[1]
        counter+=1
        makeMonthPlotData(lonValue, r'E:\lisnd16', r'E:\year plot data\year plot data ' + i +  ' 2016.csv', monthData)
        
#Loops through the files created by makeYearPlotData and averages the data by longitude
#Outputs one TEC value per bin, and assigns the average value to all points within that bin
#Also creates one file per month detailing the amount of points per bin per day
def averageYearData(hour):
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    try:
        os.mkdir(fr'E:\year plot data\averaged year data\year plot data {hour}UT')
        print("data directory created")
        
    except:
        print("directory exists")
    try:
        os.mkdir(fr'E:\year plot data\year data counts\{hour}UT')
        print("counts directory created")
    except:
        print("directory exists")
    for i in months:
        print(i)
        data = ddf.read_csv(r'E:\year plot data\unaveraged year data\year plot data ' + i +  ' 2016.csv', names=["day", "latitude", "TEC", "hour"])
        constrained = data.loc[data["hour"] == hour].compute()
        print("averaging")
        dump = averageInDegree(constrained)
        path = fr'E:\year plot data\averaged year data\year plot data {hour}UT\year plot data {i} averaged 2016.csv'
        countsPath = fr'E:\year plot data\year data counts\{hour}UT\year plot data {i} counts 2016.csv'
        createFile(dump[0], path)
        createFile(dump[1], countsPath)
        print("created files: \n" + path + "\n" + countsPath)

#Makes one csv containing all the WAM-IPE data for a given hour and longitude
def ncYearPlotData(lonValue, path, pathToFile, hourParam):
    files = []
    filenames = []
    counter = 0
    days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    for entry in os.scandir(path):
        if entry.path.endswith(".nc") and entry.is_file():
            files.append(entry.path)
            filenames.append(entry.name)
    print("directory scanned")
    for item in files:
        day = filenames[counter][8:12]
        hour = int(filenames[counter][13:15])
        year = int(filenames[counter][4:8])
        dayNum = int(day[2:])
        month = int(day[:2])
        cut = pd.DataFrame(columns = [["lat", "tec", "day"]])
        
        if hour == hourParam and year == 2016:
            print("reading: " + item)
            j = read(item)
            z = (j.loc[(j["lon"] > lonValue-1) & (j["lon"] < lonValue+1)])
            cut = z[["lat", "tec"]]
            cut["day"] = dayNum + sum(days[:month])
            if len(cut.index) != 0:
                cut.to_csv(pathToFile, header=False, mode='a')
        counter += 1
    print("file created")

#Graphs the averaged LISN data, input is the path to the directory where the files are stored
def graphYearData(path):
    paths = []
    for entry in os.scandir(path):
        if entry.path.endswith(".csv") and entry.is_file():
            paths.append(entry.path)
    for i in paths:
        data = ddf.read_csv(i).compute()
        data2 = data.loc[(data["latitude"] >= -40) & (data["latitude"] <= 20)]
        dump = graphDfNoColorbar(data2, "day", "latitude", "TEC")
    fig1.colorbar(dump[0], cax=dump[1], orientation='vertical', ticks=[0, 70])


#Combines the counts files outputted by averageYearData
def makeYearCounts(path, output_file):
    fileList = []
    for entry in os.scandir(path):
        if entry.path.endswith(".csv") and entry.is_file():
            fileList.append(entry.path)
    combined = pd.concat([pd.read_csv(f) for f in fileList])
    combined.to_csv(output_file)


#Graphs the ombined counts file outputted by makeYearCounts
def graphYearCounts(path):
    df = pd.read_csv(path, names = ["latitude", "counts", "day"], skiprows = 2)            
    sc1 = ax.hexbin(df["day"], df["latitude"], C=df["counts"], cmap=plt.cm.get_cmap("RdYlBu_r"))
    divider = make_axes_locatable(ax)
    caxes = divider.append_axes('right', size='2%', pad=0.1)
    vmin, vmax = sc1.get_clim()
    fig1.colorbar(sc1, cax=caxes, orientation='vertical', ticks=[1, 1000, 2000, 3000, 4000, 5000, 6000, vmax])
    

#Graphs the WAM-IPE data on the top subplot and the Kp and F10.7 indices on the bottom subplot
def graphIndices(modelData, indexData):
    ax.set_xlabel("Day")
    ax2.set_xlabel("Day")
    ax.set_ylabel("Latitude")
    ax2.set_ylabel("Kp index", color = "red")
    ax3 = ax2.twinx()
    ax3.set_ylabel("F10,7 Index", color = "blue")
    ax.set_title("Top: WAM-IPE 20UT 2016 | Bottom: Kp and F10.7 index average per day")
    indexData["Kp index"]/=10
    ax.margins(x=0, y=0)
    #ax.hexbin(modelData["day"], modelData["lat"], C=modelData["tec"], vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    scatterDfNoColorbar(modelData, "day", "lat", "tec")
    ax2.margins(x=0, y=0)
    ax2.plot(indexData["day"], indexData["Kp index"], color = "red")
    ax3.plot(indexData["day"], indexData["F10.7 index"], color = "blue")

#Graphs day vs TEC for both LISN and WAM-IPE on the same plot
def graphDifferences(lat, pathToLisn, pathToModel):
    counter = 0
    lisnFiles = []
    for entry in os.scandir(pathToLisn):
        if entry.path.endswith(".csv") and entry.is_file():
            lisnFiles.append(entry.path)
    for i in lisnFiles:
        print(i)
        df = pd.read_csv(i)
        lisnData = df.loc[(df["latitude"] > lat-1) & (df["latitude"] < lat+1)]
        if counter == 0:        
            ax.scatter(lisnData["latitude"], lisnData["TEC"], c="red", s=10, label = "LISN Data")
        else:
            ax.scatter(lisnData["latitude"], lisnData["TEC"], c="red", s=10)
        counter+=1
    model = pd.read_csv(pathToModel, names=["lat", "tec", "day"])
    modelData = model.loc[(model["lat"] > lat-1) & (model["lat"] < lat+1)]
    
    ax.scatter(modelData["lat"], modelData["tec"], color="blue", s=10, label = "WAM-IPE Data")
    ax.set_title(f"WAM-IPE TEC vs LISN TEC at -70 longitude and {lat} latitude")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("TEC")
    ax.legend(bbox_to_anchor=(1.15, 1.03), fancybox=True, shadow=True)
   
#Calculates root mean square error
def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())

#Graphs LISN TEC (x-axis) vs WAM-IPE TEC (y-axis) for a given latitude
def tecVsTec(lat, pathToLisn, pathToModel):
    lisnFiles = []
    lisnData = pd.DataFrame()
    for entry in os.scandir(pathToLisn):
        if entry.path.endswith(".csv") and entry.is_file():
            lisnFiles.append(entry.path)
    for i in lisnFiles:
        print(i)
        df = pd.read_csv(i)
        q = df.loc[(df["latitude"] > lat-1) & (df["latitude"] < lat+1)]
        lisnData = lisnData.append(q)
    prelim = lisnData.groupby('day').first().reset_index()
    xlist = prelim['TEC']
    days = prelim['day']
    model = pd.read_csv(pathToModel, names=["lat", "tec", "day"])
    modelData = model.loc[(model["lat"] > lat-1) & (model["lat"] < lat+1)]
    modelData = modelData.loc[modelData["day"].isin(days)]
    ylist = modelData["tec"]
    r = rmse(ylist.to_numpy(), xlist.to_numpy())
    print(ylist.to_numpy() - xlist.to_numpy())
    ax.plot([0, max(xlist)], [0, max(xlist)], color='grey', label = f"RMSE: {r}")
    ax.scatter(xlist, ylist, s=10)
    ax.set_title(f"WAM-IPE TEC vs LISN TEC at -70 longitude and {lat} latitude, 20UT")
    ax.set_xlabel("LISN Data TEC")
    ax.set_ylabel("WAM-IPE TEC")
    ax.legend(bbox_to_anchor=(1.03, 0.15), fancybox=True, shadow=True)

#Graphs latitude (x-axis) vs TEC (y-axis) for both LISN and WAM-IPE
def latVsTec(pathToLisn, pathToModel):
    lisnFiles = []
    lisnData = pd.DataFrame()
    for entry in os.scandir(pathToLisn):
        if (entry.path.endswith("March averaged 2016.csv") or entry.path.endswith("September averaged 2016.csv")) and entry.is_file():
            lisnFiles.append(entry.path)
    for i in lisnFiles:
        print(i)
        df = pd.read_csv(i)
        lisnData = lisnData.append(df)
    prelim = lisnData.groupby('TEC').first().reset_index()
    print(prelim.groupby("latitude").first().reset_index())
    prelim = prelim.loc[(prelim["latitude"] >= -50) & (prelim["latitude"] <= 30)]
    ax.scatter(prelim["latitude"], prelim["TEC"], s=1, c='red', label="LISN Data")
    model = pd.read_csv(pathToModel, names=["lat", "tec", "day"])
    modelData = model
    ax.set_title("WAM-IPE TEC vs LISN TEC at -70 longitude and latitude")
    ax.set_xlabel("LISN Data TEC")
    ax.set_ylabel("WAM-IPE TEC")
    days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    MarchStart = sum(days[:3])+1
    MarchEnd = sum(days[:4])
    JunStart = sum(days[:6])+1
    JunEnd = sum(days[:7])
    SeptemberStart = sum(days[:9])+1
    SeptemberEnd = sum(days[:10])
    DecStart = sum(days)
    DecEnd = 366
    modelData = modelData.loc[((modelData["day"] >= MarchStart) & (modelData["day"] <= MarchEnd)) | ((modelData["day"] >= SeptemberStart) & (modelData["day"] <= SeptemberEnd))]
    modelData = modelData.loc[(modelData["lat"] >= -50) & (modelData["lat"] <= 30)]
    print(modelData)
    ax.scatter(modelData["lat"], modelData["tec"], color="blue", s=1, label = "WAM-IPE Data")
    ax.set_title("WAM-IPE vs LISN TEC, -70 longitude 20UT, December")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("TEC")
    ax.legend(bbox_to_anchor=(1.15, 1.03), fancybox=True, shadow=True)
    
    
    
    #[(modelData["day"] >= JunStart) & (modelData["day"] <= JunEnd)]#
    # or entry.path.endswith("September averaged 2016.csv"))
    
#Graphs the mean of the LISN TEC on the top subplot and the mean for WAM-IPE on the bottom subplot
#Also graphs standard deviation lines below and above both mean lines
def latVsTecMeans(pathToLisn, pathToModel):
    lisnFiles = []
    lisnData = pd.DataFrame()
    ax.set_ylim(0,50)
    ax2.set_ylim(0,50)
    for entry in os.scandir(pathToLisn):
        if (entry.path.endswith("December averaged 2016.csv")):# or (entry.path.endswith("September averaged 2016.csv"))) and entry.is_file():
            lisnFiles.append(entry.path)
    for i in lisnFiles:
        print(i)
        df = pd.read_csv(i)
        lisnData = lisnData.append(df)
    prelim = lisnData.groupby('TEC').first().reset_index()
    prelim = prelim.loc[(prelim["latitude"] >= -50) & (prelim["latitude"] <= 30)]
    mid = prelim.groupby(pd.cut(prelim["latitude"], bins=80))
    final = pd.DataFrame()
    xarr = []
    for q in range(-50, 30):
        xarr.append(q)
    xarr = np.array(xarr)
    stdevarrAbove = np.array(0)
    stdevarrBelow = np.array(0)
    continues = []
    continueCount = 0
    for i in mid.groups.keys():
        try:
            data = mid.get_group(i)
            stdevs = mid.get_group(i)
            continueCount+=1
            
        except:
            continues.append(continueCount)
            continueCount+=1
            continue
        meanTEC = mean(data["TEC"])
        data["TEC"] = meanTEC
        final = final.append(data)
        stdev = np.std(stdevs["TEC"])
        stdevarrAbove = np.append(stdevarrAbove, meanTEC+stdev)
        stdevarrBelow = np.append(stdevarrBelow, meanTEC-stdev)
    xarr = np.delete(xarr, continues)
    ax.plot(final["latitude"], final["TEC"], c='red', label="LISN Data")
    ax.plot(xarr, stdevarrAbove[1:], "--", c='gray', label="LISN Data standard deviation")
    ax.plot(xarr, stdevarrBelow[1:], "--", c='gray')
    model = pd.read_csv(pathToModel, names=["lat", "tec", "day"])
    modelData = model
    days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    MarchStart = sum(days[:3])+1
    MarchEnd = sum(days[:4])
    JunStart = sum(days[:6])+1
    JunEnd = sum(days[:7])
    SeptemberStart = sum(days[:9])+1
    SeptemberEnd = sum(days[:10])
    DecStart = sum(days)
    DecEnd = 366
    modelData = modelData.loc[((modelData["day"] >= JunStart) & (modelData["day"] <= JunEnd))]# | ((modelData["day"] >= SeptemberStart) & (modelData["day"] <= SeptemberEnd))]
    modelData = modelData.loc[(modelData["lat"] >= -50) & (modelData["lat"] <= 30)]
    modelData2 = modelData.groupby("lat")
    final2 = pd.DataFrame()
    xarr2 = modelData["lat"].unique()
    stdevarrAbove2 = np.array(0)
    stdevarrBelow2 = np.array(0)
    
    for i in modelData2.groups.keys():
        data2 = modelData2.get_group(i)
        stdevs2 = modelData2.get_group(i)
        mean2 = mean(data2["tec"])
        data2["tec"] = mean2
        final2 = final2.append(data2)
        stdev2 = np.std(stdevs2["tec"])
        stdevarrAbove2 = np.append(stdevarrAbove2, mean2+stdev2)
        stdevarrBelow2 = np.append(stdevarrBelow2, mean2-stdev2)
    ax2.plot(final2["lat"], final2["tec"], color="blue", label = "WAM-IPE Data")
    ax2.plot(xarr2, stdevarrAbove2[1:], "--", c='green', label="WAM-IPE standard deviation")
    ax2.plot(xarr2, stdevarrBelow2[1:], "--", c='green') 
    ax.set_title("Top: LISN TEC | Bottom: WAM-IPE TEC, -70 longitude 20UT, December median lines", fontsize = 10)
    ax2.set_xlabel("Latitude")
    ax.set_ylabel("TEC")
    ax2.set_ylabel("TEC")
    fig1.legend(bbox_to_anchor=(1.35, 0.9), fancybox=True, shadow=True)
    
#Graphs the LISN data on the top subplot and graphs the Kp nd F10.7 ind
def graphLisnVsIndices(pathToLisn, indexData):
    ax.set_xlabel("Day")
    ax2.set_xlabel("Day")
    ax.set_ylabel("Latitude")
    ax2.set_ylabel("Kp index", color = "red")
    ax3 = ax2.twinx()
    ax3.set_ylabel("F10,7 Index", color = "blue")
    ax.set_title("Top: LISN 20UT 2016 | Bottom: Kp and F10.7 index average per day")
    indexData["Kp index"]/=10
    ax.margins(x=0, y=0)
    #ax.hexbin(modelData["day"], modelData["lat"], C=modelData["tec"], vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    graphYearData(pathToLisn)
    ax2.margins(x=0, y=0)
    ax2.plot(indexData["day"], indexData["Kp index"], color = "red")
    ax3.plot(indexData["day"], indexData["F10.7 index"], color = "blue")

#Graphs the LISN data on the top subplot and the WAM-IPE data on the bottom subplot
def graphLisnVsModel(pathToLisn, modelData):
    ax.set_xlabel("Day")
    ax2.set_xlabel("Day")
    ax.set_ylabel("Latitude")
    ax2.set_ylabel("Latitude")
    ax.set_title("Top: LISN 20UT 2016 | Bottom: WAM-IPE 20UT 2016")
    ax.margins(x=0, y=0)
    #ax.hexbin(modelData["day"], modelData["lat"], C=modelData["tec"], vmin = 0, vmax = 70, cmap=plt.cm.get_cmap("RdYlBu_r"))
    graphYearData(pathToLisn)
    ax2.margins(x=0, y=0)
    scatterDf(modelData, "day", "lat", "tec")
