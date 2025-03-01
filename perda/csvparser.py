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
