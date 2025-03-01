import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_csv(logfile, value_map, hvChanges, infer_timestamps=False):
  log = open(logfile, 'r')
  lines = log.readlines()

  key_map = dict() # id -> name

  startTime = 0
  timeDelta = 0
  hvOn = False

  for line in lines[1:]:
    try:
      if line.startswith("Value"):
        s = line[6:].strip().split(": ")
        key_map[int(s[1])] = s[0]
      else:
        s = line.strip().split(",")

        id = int(s[1])
        val = float(s[2])

        name = key_map[id]

        if "sdl.startTime" in name:
          startTime = val
        elif "sdl.currentTime" in name:
          timeDelta = val / 1e6

        timestamp = timeDelta

        if "ams.airsState" in name:
          if hvOn and val == 0:
            hvOn = False
            hvChanges.append((timestamp, hvOn))
          elif not hvOn and val == 4:
            hvOn = True
            hvChanges.append((timestamp, hvOn))

        if name not in value_map:
          value_map[name] = list()

        if infer_timestamps:
          if len(value_map[name]) == 0:
            timestamp = 0
          else:
            timestamp = value_map[name][len(value_map[name]) - 1][0] + 1

        value_map[name].append(list([timestamp, val, hvOn]))
    except:
      print("Error parsing line: " + line)
      continue

def plot(variables, value_map, hvChanges, same_graph=False):
  for var in variables:
    if type(var) is list:
      plot(var, value_map, hvChanges, True)
      continue

    short_name = var[0] if type(var) is tuple else var

    full_name = None

    for var_name in value_map.keys():
      if short_name in var_name:
        full_name = var_name
        break

    if full_name is None:
      print("Error: could not find data for " + short_name)
      continue

    vals = np.array(value_map[full_name])
    t = vals[::, 0]
    y = vals[::, 1]

    if type(var) is tuple:
      f = var[1]
      y = [f(n) for n in y]

    plt.plot(t, y, label=short_name)
    plt.xlabel("Timestamp (s)")
    plt.legend()

    for hvT, hvOn in hvChanges:
      color = "red" if hvOn else "green"
      plt.axvline(x=hvT, color=color)

    if not same_graph:
      plt.title(full_name)
      plt.show()
      print("\n\n")

  if same_graph:
    plt.show()
    print("\n\n")

def calculate0to60(value_map):
  short_name = "pcm.wheelSpeeds.frontLeft"

  full_name = None

  for var_name in value_map.keys():
    if short_name in var_name:
      full_name = var_name
      break

  if full_name is None:
    print("ERROR")
    return

  minTime = (float('inf'), -1, -1, -1) # duration, startTime, endTime, endSpeed
  startTime = -1
  endTime = -1

  vals = np.array(value_map[full_name])

  lastTime = -1
  lastSpeed = -1

  for t, y, hv in vals:
    # wheel speed sensor is NaN if 0
    if np.isnan(y):
      startTime = t
    elif y >= 60 and lastSpeed < 60:
      endTime = lastTime + (60 - lastSpeed) * (t - lastTime) / (y - lastSpeed)

      duration = endTime - startTime

      if duration < minTime[0]:
        minTime = (duration, startTime, t, y)

    lastTime = t
    lastSpeed = y

  if minTime[1] == -1:
    print("DID NOT REACH 60 MPH")
    return

  t = vals[:, 0]
  y = vals[:, 1]

  mask = (t >= minTime[1]) & (t <= minTime[2])
  t = t[mask]
  y = y[mask]

  plt.plot(t, y)
  plt.axhline(y=60, color='red')
  plt.xlabel("Timestamp (s)")
  plt.title("0-60 MPH in " + str(minTime[0]) + "s")
  plt.show()