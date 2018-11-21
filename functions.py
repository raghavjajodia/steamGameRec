import requests
import time 

def getOwnedGames(steamid, include_appinfo = True, include_played_free_games = True, appids_filter = False) :
    '''
    Return json/dict of list of games owned by the player
    Output format (default args) : {'game_count' : xx, 'games' : [{'appid' : xx, 'playtime_forever' : xx}]}
    Assumes key is in file "key.txt"

    Reference : https://partner.steamgames.com/doc/webapi/IPlayerService
    '''
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    
    try :
        with open("key.txt", "r") as f :
            key = f.read()
    except IOerror :
        print("key.txt not found")
        return(None)

    key = key.strip("\n")

    params = {"key" : key, "steamid" : steamid, "include_appinfo" : include_appinfo, 
                "include_played_free_games"  : include_played_free_games, "appids_filter" : appids_filter}

    req = requests.get(url, params = params)
    return(req.json()["response"])