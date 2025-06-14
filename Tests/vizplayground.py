# Old code that needs to be updated
# -*- coding: utf-8 -*-
"""VizPlayground.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/112dslnp5Kwgb9RxHiQTSsbNvCb1VAsMb

# Penn Electric Racing Car Data Processor PerCan Data Processor+++ v2.0 (PERCDPPCDP+++ v2.0)

Welcome to the PERCDPPCDP++ v2.0: your home for data processing needs.

Instructions:
- This code mounts google drive in order to retrieve log data directly from FSAE Penn Electric Racing Drive.
If you decide to use google drive, you can simply edit the folder path in `folder_path` to use testing data from other cars. You can edit the `logfile` to specify the file path.
- If not using google drive, upload log to Colab and update the name in `logfile`
-Find or create github authentication token at https://github.com/settings/tokens for access to the library
- Enter the names of what you want graphed under `variables`
  - Examples:
    - Just a variable name: `pcm.moc.motor.wheelSpeed`
    - Can use tuples to calculate lambda transformation functions: `("pcm.wheelSpeeds.backLeft", lambda x: 2.5/3.5*x)`
    - Graph multiple variables on the same graph by storing them as a list: `["pcm.wheelSpeeds.backLeft", "pcm.wheelSpeeds.backRight"]`
- You can add calls to `print_stats(short_name, value_map, requireHv)`
- Run all

Note: Red vertical lines indicate HV turned on and green vertical lines indicate HV turned off.
"""

from google.colab import drive
drive.mount('/content/drive')

'''path to folder --> keep only one of the next two lines depending on if
using personal google account or logged into PER account '''
#folder_path = '/content/drive/MyDrive/REV8/TESTING/REV8 Testing Data/'
folder_path = '/content/drive/Shareddrives/FSAE Penn Electric Racing/REV8/TESTING/REV8 Testing Data'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import struct
import tqdm
import os

#run this cell if you need to see the available directories
os.listdir(folder_path)

# Name of the log file
logfile = folder_path + "/10_08/01_01_00 12_18_59 AM.csv"
#if not using google drive
#logfile = "/04_24/04_24_23 07_41_13 PM.csv"

#Configuration to access library
#Input your github username and authentication token
username = ""
authenticationToken = ""
!pip uninstall -y perda
!pip install --upgrade git+https://{username}:{authenticationToken}@github.com/Penn-Electric-Racing/PER-Data-Analyzer.git
# You may need to restart session if updates are done on library in real time. Run this cell then restart session

from perda import *
aly = create() #creating analyser object

aly.reset() #resetting analyser object before reading new csv
aly.read_csv(logfile, "sdl.currentTime") #reading the csv file

variables = ["pcm.wheelSpeeds.frontRight",
             "pcm.wheelSpeeds.frontLeft",
]

aly.analyze_data(variables)
aly.analyze_data(variables, start_time=500, end_time=600, unit="s")
aly.analyze_data(variables, start_time=0, end_time=1, unit="s")

# Variables to graph
variables = [
    "pcm.moc.motor.requestedTorque",
    "pcm.moc.motor.wheelSpeed",
    "ams.pack.power",
    "ams.pack.voltage"
]
aly.set_plot(start_time = 450000, end_time = 608000, unit = "ms")
plt.axhline(y=0)
aly.plot(variables)

v_op = [
    "ams.pack.voltage",
    "*",
    "ams.pack.current"
]

power = aly.get_compute_arrays(v_op, match_type="connect")

variables = [
    power
]

aly.set_plot(start_time = 601, end_time = 608, unit = "s")
aly.plot(variables)

variables = [
    "ams.pack.power"
]

aly.set_plot(start_time = 601000, end_time = 608000, unit = "ms")
plt.axhline(y=0)  # Vertical line at x=2
aly.plot(variables)

# Variables to graph
variables = ["pcm.wheelSpeeds.frontLeft",
             "pcm.wheelSpeeds.frontRight",
             "pcm.wheelSpeeds.backLeft",
             "pcm.wheelSpeeds.backRight"]
aly.set_plot()
plt.axhline(y=0)
aly.plot(variables)

aly.calculate0to60(4)