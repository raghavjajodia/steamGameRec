import json
from functions import getOwnedGames

#Getting SteamIds
steamids = []
with open("data/SteamIds/steamIds_1.8mi.txt", "r") as f :
    for steamid in f :
        steamids.append(steamid.strip("\n"))

#Keeping track of number of SteamIds done
with open("config.txt", "r") as f :
    config = eval(f.read())

current = config["current_id_loc"]
stop = config["stop_id_loc"]

#Opening ownedgames.json and updating the data dictionary to not lose progress
with open("data/ownedgames.json", "r") as f :
    data = json.loads(f.read())

#Start from current and end at stop; Update current to stop
print("Starting from {}, ending at {}".format(current, stop))
for i in range(current, stop, 1) :
    print("Progress : {0:.2f}%".format((i - current) * 100 / (stop - current)), end = "\r")
    steamid = steamids[i]
    data[steamid] = getOwnedGames(steamid)

#Update config
config["current_id_loc"] = stop
config["stop_id_loc"] = min(stop + 10000, len(steamids))
with open("config.txt", "w") as f :
    f.write(str(config))

#Dump data to ownedgames.json (Overwrites)
with open("data/ownedgames.json", "w") as f :
    json.dump(data, f)