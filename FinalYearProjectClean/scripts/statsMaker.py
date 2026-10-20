import pandas as pd
from pathlib import Path
from collections import deque

from datetime import datetime

import numpy as np


# this is a storage for stats so that i can calclate pre fight stats for each fight
def createFighter():
    return {
        "fights": 0,
        "totalTimeSec": 0,
        "kd": 0,
        "sigLanded": 0,
        "sigAttempted": 0,
        "sigAbsorbed": 0,
        "sigFaced": 0,
        "tdLanded": 0,
        "tdAttempted": 0,
        "tdAllowed": 0,
        "tdFaced": 0,
        "subAttempts": 0,
        "reversals": 0,
        "controlTimeSec": 0,
        "wins": 0,
        "finishes": 0,
        "lastFightDate": None,
        "currentWinStreak": 0,
        "recent": deque(maxlen=3)
    }

# path for the csv
basePath = Path(__file__).resolve().parents[1]
dataPath = basePath / "data"

csvPath = dataPath / "merged_stats_n_scorecards_reversed.csv"
fightersPath = dataPath / "fighters_clean.csv"



fightersDf = pd.read_csv(fightersPath)

df = pd.read_csv(csvPath)



fightersLookup = fightersDf.set_index("fighter_name").to_dict("index")

# the storage for fighter stats and then the holder for final training csv
fighters = {}
trainingRows = []

# the first 204 fights use the old ufc system of one round so data is different and will add noise
startIndex = 204


def getFightTime(round, time):
      finalTime = (int(round) - 1) * 300
      minutes, seconds = time.split(":")
      totalSeconds = int(minutes) * 60 + int(seconds)
      finalTime = finalTime + totalSeconds
      return finalTime

#use this for landed and absorbed takedowns etc
def ofLeft(dataPoint):
      left, right = map(int, dataPoint.split(" of "))
      return left

def ofRight(dataPoint):
      left, right = map(int, dataPoint.split(" of "))
      return right

def timeConvert(time):
      if not time or ":" not in str(time):
            return 0
      minutes, seconds = str(time).split(":")
      return int(minutes) * 60 + int(seconds)


def heightToCm(h):
    if not h or not isinstance(h, str):
        return 0
    f, i = h.replace('"', '').split("'")
    return int((int(f)*12 + int(i)) * 2.54)

def reachToCm(s):
    if not s or not isinstance(s, str):
        return 0
    inches = int(s.replace('"', ''))
    return int(inches * 2.54)


def ageFromDob(dob_str, date_str):
    # guard against NaN / blank
    if not dob_str or not isinstance(dob_str, str):
        return 0
    if not date_str or not isinstance(date_str, str):
        return 0

    dob = datetime.strptime(dob_str, "%b %d, %Y")
    current = datetime.strptime(date_str, "%d/%m/%Y")

    age = current.year - dob.year
    if (current.month, current.day) < (dob.month, dob.day):
        age -= 1

    return age

def daysSince(lastDate, fightDate):
    if not lastDate:
        return 0
    d1 = datetime.strptime(lastDate, "%d/%m/%Y")
    d2 = datetime.strptime(fightDate, "%d/%m/%Y")
    return (d2 - d1).days




weightMap = {
    "Strawweight": 0,
    "Flyweight": 1,
    "Bantamweight": 2,
    "Featherweight": 3,
    "Lightweight": 4,
    "Welterweight": 5,
    "Middleweight": 6,
    "Light Heavyweight": 7,
    "Heavyweight": 8
}

def parseBoutType(bout):
    if not isinstance(bout, str):
        return 0, 0, 0

    boutLower = bout.lower()

    isTitle = 1 if "title" in boutLower else 0
    isWomens = 1 if "women" in boutLower else 0

    weightClass = 0
    for name, val in weightMap.items():
        if name.lower() in boutLower:
            weightClass = val
            break

    return weightClass, isTitle, isWomens


def parseRounds(time_format):
    if not isinstance(time_format, str):
        return 0
    try:
        return int(time_format.split(" ")[0])
    except:
        return 0








for i in range(startIndex, len(df)):

    row = df.iloc[i]

    method = str(row["method"]).lower()

    redName = row["red_fighter_name"]
    blueName = row["blue_fighter_name"]

    if redName not in fighters:
        fighters[redName] = createFighter()

    if blueName not in fighters:
        fighters[blueName] = createFighter()

    redStats = fighters[redName]
    blueStats = fighters[blueName]

    redReady = redStats["fights"] >= 3
    blueReady = blueStats["fights"] >= 3

    

    if redReady and blueReady:

        newTrainingRow = {}

        redInfo = fightersLookup.get(redName, {})
        blueInfo = fightersLookup.get(blueName, {})



        # new features for MK3
        weightClass, isTitle, isWomens = parseBoutType(row["bout_type"])
        maxRounds = parseRounds(row["time_format"])

        newTrainingRow["weightClass"] = weightClass
        newTrainingRow["isTitleFight"] = isTitle
        newTrainingRow["isWomensFight"] = isWomens
        newTrainingRow["maxRounds"] = maxRounds

        # ----------- 
        # RED Section - Do all of reds stats first 
        # -----------

        # first the personal stats at time of fight

        redHeight = redInfo.get("Height", "")
        redReach = redInfo.get("Reach", "")
        redStance = redInfo.get("Stance", "")
        redDob = redInfo.get("DOB", "")

        newTrainingRow["redName"] = redName
        newTrainingRow["redHeightCm"] = heightToCm(redHeight)
        newTrainingRow["redReachCm"] = reachToCm(redReach) 
        newTrainingRow["redStance"] = redStance if redStance else 0
        newTrainingRow["redAge"] = ageFromDob(redDob, row["event_date"]) 

        # average of of stats over last fights

        redMinutes = redStats["totalTimeSec"] / 60
        redFights = redStats["fights"]

        redKdPerMin = redStats["kd"] / redMinutes if redMinutes else 0
        redSigLandedPerMin = redStats["sigLanded"] / redMinutes if redMinutes else 0
        redSigThrownPerMin = redStats["sigAttempted"] / redMinutes if redMinutes else 0
        redSigAccuracy = redStats["sigLanded"] / redStats["sigAttempted"] if redStats["sigAttempted"] else 0
        redSigAbsorbedPerMin = redStats["sigAbsorbed"] / redMinutes if redMinutes else 0
        redSigDefense = 1 - (redStats["sigAbsorbed"] / redStats["sigFaced"]) if redStats["sigFaced"] else 0

        redTdPerMin = redStats["tdLanded"] / redMinutes if redMinutes else 0
        redTdAccuracy = redStats["tdLanded"] / redStats["tdAttempted"] if redStats["tdAttempted"] else 0
        redTdDefense = 1 - (redStats["tdAllowed"] / redStats["tdFaced"]) if redStats["tdFaced"] else 0

        redSubPerMin = redStats["subAttempts"] / redMinutes if redMinutes else 0
        redRevPerMin = redStats["reversals"] / redMinutes if redMinutes else 0
        redControlRatio = redStats["controlTimeSec"] / redStats["totalTimeSec"] if redStats["totalTimeSec"] else 0

        redWinRate = redStats["wins"] / redFights if redFights else 0
        redFinishRate = redStats["finishes"] / redStats["wins"] if redStats["wins"] else 0

        redRecentWins = sum(f["win"] for f in redStats["recent"])
        redRecentMinutes = sum(f["minutes"] for f in redStats["recent"])
        redRecentSig = sum(f["sigLanded"] for f in redStats["recent"])
        redRecentKd = sum(f["kd"] for f in redStats["recent"])
        redRecentTd = sum(f["td"] for f in redStats["recent"])

        redRecentWinRate = redRecentWins / len(redStats["recent"]) if redStats["recent"] else 0
        redRecentSigPerMin = redRecentSig / redRecentMinutes if redRecentMinutes else 0
        redRecentKdPerMin = redRecentKd / redRecentMinutes if redRecentMinutes else 0
        redRecentTdPerMin = redRecentTd / redRecentMinutes if redRecentMinutes else 0

        redDaysSinceLast = daysSince(redStats["lastFightDate"], row["event_date"])
        redAvgFightTime = redMinutes / redFights if redFights else 0

        newTrainingRow["redKdPerMin"] = redKdPerMin
        newTrainingRow["redSigLandedPerMin"] = redSigLandedPerMin
        newTrainingRow["redSigThrownPerMin"] = redSigThrownPerMin
        newTrainingRow["redSigAccuracy"] = redSigAccuracy
        newTrainingRow["redSigAbsorbedPerMin"] = redSigAbsorbedPerMin
        newTrainingRow["redSigDefense"] = redSigDefense
        newTrainingRow["redTdPerMin"] = redTdPerMin
        newTrainingRow["redTdAccuracy"] = redTdAccuracy
        newTrainingRow["redTdDefense"] = redTdDefense
        newTrainingRow["redSubPerMin"] = redSubPerMin
        newTrainingRow["redRevPerMin"] = redRevPerMin
        newTrainingRow["redControlRatio"] = redControlRatio
        newTrainingRow["redWinRate"] = redWinRate
        newTrainingRow["redRecentWinRate"] = redRecentWinRate
        newTrainingRow["redTotalFights"] = redFights
        newTrainingRow["redDaysSinceLastFight"] = redDaysSinceLast
        newTrainingRow["redFinishRate"] = redFinishRate
        newTrainingRow["redRecentSigPerMin"] = redRecentSigPerMin
        newTrainingRow["redRecentKdPerMin"] = redRecentKdPerMin
        newTrainingRow["redRecentTdPerMin"] = redRecentTdPerMin
        newTrainingRow["redCareerMinutes"] = redMinutes
        newTrainingRow["redAvgFightTime"] = redAvgFightTime
        newTrainingRow["redWinStreak"] = redStats["currentWinStreak"]


        # -----------
        # BLUE Section - once all of red personal and pre fights stats are done add blues 
        # -----------

        # first the personal stats at time of fight

        blueHeight = blueInfo.get("Height", "")
        blueReach = blueInfo.get("Reach", "")
        blueStance = blueInfo.get("Stance", "")
        blueDob = blueInfo.get("DOB", "")

        newTrainingRow["blueName"] = blueName
        newTrainingRow["blueHeightCm"] = heightToCm(blueHeight)
        newTrainingRow["blueReachCm"] = reachToCm(blueReach) 
        newTrainingRow["blueStance"] = blueStance if blueStance else 0
        newTrainingRow["blueAge"] = ageFromDob(blueDob, row["event_date"]) 

        # average of of stats over last fights

        blueMinutes = blueStats["totalTimeSec"] / 60
        blueFights = blueStats["fights"]

        blueKdPerMin = blueStats["kd"] / blueMinutes if blueMinutes else 0
        blueSigLandedPerMin = blueStats["sigLanded"] / blueMinutes if blueMinutes else 0
        blueSigThrownPerMin = blueStats["sigAttempted"] / blueMinutes if blueMinutes else 0
        blueSigAccuracy = blueStats["sigLanded"] / blueStats["sigAttempted"] if blueStats["sigAttempted"] else 0
        blueSigAbsorbedPerMin = blueStats["sigAbsorbed"] / blueMinutes if blueMinutes else 0
        blueSigDefense = 1 - (blueStats["sigAbsorbed"] / blueStats["sigFaced"]) if blueStats["sigFaced"] else 0

        blueTdPerMin = blueStats["tdLanded"] / blueMinutes if blueMinutes else 0
        blueTdAccuracy = blueStats["tdLanded"] / blueStats["tdAttempted"] if blueStats["tdAttempted"] else 0
        blueTdDefense = 1 - (blueStats["tdAllowed"] / blueStats["tdFaced"]) if blueStats["tdFaced"] else 0

        blueSubPerMin = blueStats["subAttempts"] / blueMinutes if blueMinutes else 0
        blueRevPerMin = blueStats["reversals"] / blueMinutes if blueMinutes else 0
        blueControlRatio = blueStats["controlTimeSec"] / blueStats["totalTimeSec"] if blueStats["totalTimeSec"] else 0

        blueWinRate = blueStats["wins"] / blueFights if blueFights else 0
        blueFinishRate = blueStats["finishes"] / blueStats["wins"] if blueStats["wins"] else 0

        blueRecentWins = sum(f["win"] for f in blueStats["recent"])
        blueRecentMinutes = sum(f["minutes"] for f in blueStats["recent"])
        blueRecentSig = sum(f["sigLanded"] for f in blueStats["recent"])
        blueRecentKd = sum(f["kd"] for f in blueStats["recent"])
        blueRecentTd = sum(f["td"] for f in blueStats["recent"])

        blueRecentWinRate = blueRecentWins / len(blueStats["recent"]) if blueStats["recent"] else 0
        blueRecentSigPerMin = blueRecentSig / blueRecentMinutes if blueRecentMinutes else 0
        blueRecentKdPerMin = blueRecentKd / blueRecentMinutes if blueRecentMinutes else 0
        blueRecentTdPerMin = blueRecentTd / blueRecentMinutes if blueRecentMinutes else 0

        blueDaysSinceLast = daysSince(blueStats["lastFightDate"], row["event_date"])
        blueAvgFightTime = blueMinutes / blueFights if blueFights else 0

        newTrainingRow["blueKdPerMin"] = blueKdPerMin
        newTrainingRow["blueSigLandedPerMin"] = blueSigLandedPerMin
        newTrainingRow["blueSigThrownPerMin"] = blueSigThrownPerMin
        newTrainingRow["blueSigAccuracy"] = blueSigAccuracy
        newTrainingRow["blueSigAbsorbedPerMin"] = blueSigAbsorbedPerMin
        newTrainingRow["blueSigDefense"] = blueSigDefense
        newTrainingRow["blueTdPerMin"] = blueTdPerMin
        newTrainingRow["blueTdAccuracy"] = blueTdAccuracy
        newTrainingRow["blueTdDefense"] = blueTdDefense
        newTrainingRow["blueSubPerMin"] = blueSubPerMin
        newTrainingRow["blueRevPerMin"] = blueRevPerMin
        newTrainingRow["blueControlRatio"] = blueControlRatio
        newTrainingRow["blueWinRate"] = blueWinRate
        newTrainingRow["blueRecentWinRate"] = blueRecentWinRate
        newTrainingRow["blueTotalFights"] = blueFights
        newTrainingRow["blueDaysSinceLastFight"] = blueDaysSinceLast
        newTrainingRow["blueFinishRate"] = blueFinishRate
        newTrainingRow["blueRecentSigPerMin"] = blueRecentSigPerMin
        newTrainingRow["blueRecentKdPerMin"] = blueRecentKdPerMin
        newTrainingRow["blueRecentTdPerMin"] = blueRecentTdPerMin
        newTrainingRow["blueCareerMinutes"] = blueMinutes
        newTrainingRow["blueAvgFightTime"] = blueAvgFightTime
        newTrainingRow["blueWinStreak"] = blueStats["currentWinStreak"]



        # =========================
        # MK2 FEATURE ENGINEERING
        # =========================

        eps = 1e-5

        # ---------- Interaction Features ----------
        newTrainingRow["redEffectiveStriking"] = redSigLandedPerMin - blueSigAbsorbedPerMin
        newTrainingRow["blueEffectiveStriking"] = blueSigLandedPerMin - redSigAbsorbedPerMin

        newTrainingRow["redGrapplingEdge"] = redTdAccuracy * (1 - blueTdDefense)
        newTrainingRow["blueGrapplingEdge"] = blueTdAccuracy * (1 - redTdDefense)

        # ---------- Style Matchups ----------
        newTrainingRow["redStrikerVsGrappler"] = redSigLandedPerMin * blueTdPerMin
        newTrainingRow["blueStrikerVsGrappler"] = blueSigLandedPerMin * redTdPerMin

        # ---------- Threat Features ----------
        newTrainingRow["redKOThreat"] = redKdPerMin * redSigLandedPerMin
        newTrainingRow["blueKOThreat"] = blueKdPerMin * blueSigLandedPerMin

        newTrainingRow["redSubThreat"] = redSubPerMin * redTdPerMin
        newTrainingRow["blueSubThreat"] = blueSubPerMin * blueTdPerMin

        # ---------- Activity / Ring Rust ----------
        newTrainingRow["redIsRingRust"] = 1 if redDaysSinceLast > 365 else 0
        newTrainingRow["blueIsRingRust"] = 1 if blueDaysSinceLast > 365 else 0

        newTrainingRow["redActivityScore"] = 1 / (1 + redDaysSinceLast) if redDaysSinceLast else 1
        newTrainingRow["blueActivityScore"] = 1 / (1 + blueDaysSinceLast) if blueDaysSinceLast else 1

        # ---------- Nonlinear Age ----------
        newTrainingRow["redAgeSquared"] = newTrainingRow["redAge"] ** 2
        newTrainingRow["blueAgeSquared"] = newTrainingRow["blueAge"] ** 2

        # ---------- Ratios ----------
        newTrainingRow["sigLandedRatio"] = redSigLandedPerMin / (blueSigLandedPerMin + eps)
        newTrainingRow["tdRatio"] = redTdPerMin / (blueTdPerMin + eps)

        # ---------- Pace x Accuracy ----------
        newTrainingRow["redPaceXAcc"] = redSigThrownPerMin * redSigAccuracy
        newTrainingRow["bluePaceXAcc"] = blueSigThrownPerMin * blueSigAccuracy






        # ----------- 
        # Diferntials - after all the main stats are done get the differnces 
        # ----------- 
         
        numericKeys = [
            "HeightCm","ReachCm","Age",
            "KdPerMin","SigLandedPerMin","SigThrownPerMin","SigAccuracy",
            "SigAbsorbedPerMin","SigDefense",
            "TdPerMin","TdAccuracy","TdDefense",
            "SubPerMin","RevPerMin","ControlRatio",
            "WinRate","RecentWinRate","TotalFights",
            "DaysSinceLastFight","FinishRate",
            "RecentSigPerMin","RecentKdPerMin","RecentTdPerMin",
            "CareerMinutes","AvgFightTime","WinStreak",

            # MK2 features
            "EffectiveStriking","GrapplingEdge",
            "StrikerVsGrappler",
            "KOThreat","SubThreat",
            "ActivityScore","AgeSquared",
            "PaceXAcc",

            


        ]

        for key in numericKeys:
            redVal = newTrainingRow.get("red" + key, 0)
            blueVal = newTrainingRow.get("blue" + key, 0)
            newTrainingRow["diff" + key] = redVal - blueVal

        

        
        # ----------- 
        # Lables - finally say who won and how 
        # -----------

        redResult = row["red_fighter_result"]
        

        redWin = 1 if redResult == "W" else 0
        newTrainingRow["redWin"] = redWin

        newTrainingRow["winKo"] = 0
        newTrainingRow["winSub"] = 0
        newTrainingRow["winDecUnanimous"] = 0
        newTrainingRow["winDecSplit"] = 0
        newTrainingRow["winDecMajority"] = 0

        if "ko" in method or "tko" in method:
            newTrainingRow["winKo"] = 1

        elif "submission" in method:
            newTrainingRow["winSub"] = 1

        elif "decision" in method:
            if "unanimous" in method:
                newTrainingRow["winDecUnanimous"] = 1
            elif "split" in method:
                newTrainingRow["winDecSplit"] = 1
            elif "majority" in method:
                newTrainingRow["winDecMajority"] = 1

        for key, value in newTrainingRow.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                newTrainingRow[key] = float(value)


        trainingRows.append(newTrainingRow)


    # ADD FIGHTER STATS TO DICTIONARY

    fightSeconds = getFightTime(row["round"], row["time"])
    fightMinutes = fightSeconds / 60

    redSigLandedNow = int(ofLeft(row["red_fighter_sig_str"]))
    redSigAttemptedNow = int(ofRight(row["red_fighter_sig_str"]))
    redTdLandedNow = int(ofLeft(row["red_fighter_TD"]))
    redTdAttemptedNow = int(ofRight(row["red_fighter_TD"]))
    redSubNow = int(row["red_fighter_sub_att"])
    redRevNow = int(row["red_fighter_rev"])
    redKdNow = int(row["red_fighter_KD"])
    redControlNow = int(timeConvert(row["red_fighter_ctrl"]))

    blueSigLandedNow = int(ofLeft(row["blue_fighter_sig_str"]))
    blueSigAttemptedNow = int(ofRight(row["blue_fighter_sig_str"]))
    blueTdLandedNow = int(ofLeft(row["blue_fighter_TD"]))
    blueTdAttemptedNow = int(ofRight(row["blue_fighter_TD"]))
    blueSubNow = int(row["blue_fighter_sub_att"])
    blueRevNow = int(row["blue_fighter_rev"])
    blueKdNow = int(row["blue_fighter_KD"])
    blueControlNow = int(timeConvert(row["blue_fighter_ctrl"]))

    redWinNow = 1 if row["red_fighter_result"] == "W" else 0
    blueWinNow = 1 - redWinNow

    redFinishNow = 1 if redWinNow and ("ko" in method or "submission" in method) else 0
    blueFinishNow = 1 if blueWinNow and ("ko" in method or "submission" in method) else 0

    redStats["fights"] += 1
    redStats["totalTimeSec"] += fightSeconds
    redStats["kd"] += redKdNow
    redStats["sigLanded"] += redSigLandedNow
    redStats["sigAttempted"] += redSigAttemptedNow
    redStats["sigAbsorbed"] += blueSigLandedNow
    redStats["sigFaced"] += blueSigAttemptedNow
    redStats["tdLanded"] += redTdLandedNow
    redStats["tdAttempted"] += redTdAttemptedNow
    redStats["tdAllowed"] += blueTdLandedNow
    redStats["tdFaced"] += blueTdAttemptedNow
    redStats["subAttempts"] += redSubNow
    redStats["reversals"] += redRevNow
    redStats["controlTimeSec"] += redControlNow
    redStats["wins"] += redWinNow
    redStats["finishes"] += redFinishNow
    redStats["lastFightDate"] = row["event_date"]

    if redWinNow:
        redStats["currentWinStreak"] += 1
    else:
        redStats["currentWinStreak"] = 0

    redStats["recent"].append({
        "win": redWinNow,
        "minutes": fightMinutes,
        "sigLanded": redSigLandedNow,
        "kd": redKdNow,
        "td": redTdLandedNow
    })

    

    blueStats["fights"] += 1
    blueStats["totalTimeSec"] += fightSeconds
    blueStats["kd"] += blueKdNow
    blueStats["sigLanded"] += blueSigLandedNow
    blueStats["sigAttempted"] += blueSigAttemptedNow
    blueStats["sigAbsorbed"] += redSigLandedNow
    blueStats["sigFaced"] += redSigAttemptedNow
    blueStats["tdLanded"] += blueTdLandedNow
    blueStats["tdAttempted"] += blueTdAttemptedNow
    blueStats["tdAllowed"] += redTdLandedNow
    blueStats["tdFaced"] += redTdAttemptedNow
    blueStats["subAttempts"] += blueSubNow
    blueStats["reversals"] += blueRevNow
    blueStats["controlTimeSec"] += blueControlNow
    blueStats["wins"] += blueWinNow
    blueStats["finishes"] += blueFinishNow
    blueStats["lastFightDate"] = row["event_date"]

    if blueWinNow:
        blueStats["currentWinStreak"] += 1
    else:
        blueStats["currentWinStreak"] = 0

    blueStats["recent"].append({
        "win": blueWinNow,
        "minutes": fightMinutes,
        "sigLanded": blueSigLandedNow,
        "kd": blueKdNow,
        "td": blueTdLandedNow
    })


#print(trainingRows)

# -----------------------
# SAVE TRAINING DATASET
# -----------------------

trainingDf = pd.DataFrame(trainingRows)

outputPath = dataPath / "training_dataset_mk3.csv"

outputPath.parent.mkdir(parents=True, exist_ok=True)

trainingDf.to_csv(outputPath, index=False)

print(f"\nSaved training data to: {outputPath}")
print(f"Rows: {len(trainingDf)}")
print(f"Columns: {len(trainingDf.columns)}")    