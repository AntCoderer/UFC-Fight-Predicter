import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

modelPath = MODELS_DIR / "logistic_model_mk3.pkl"
featurePath = MODELS_DIR / "features.pkl"
importancePath = MODELS_DIR / "feature_importance.pkl"
scalerPath = MODELS_DIR / "scaler.pkl"
fightersPath = DATA_DIR / "fighter_profiles_mk3.csv"

weight_class_map = {
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

stance_map = {
    "Orthodox": 0,
    "Southpaw": 1,
    "Switch": 2,
    "Open Stance": 3
}


model = joblib.load(modelPath)
features = joblib.load(featurePath)
importance = joblib.load(importancePath)
scaler = joblib.load(scalerPath)

fighters_df = pd.read_csv(fightersPath)
fighters_df["stanceEncoded"] = fighters_df["stance"].map(stance_map).fillna(0)




def predict_fight(fighter_row_dict):


    df_input = pd.DataFrame([fighter_row_dict])

    for col in features:
        if col not in df_input:
            df_input[col] = 0

    df_input = df_input[features]
    df_scaled = scaler.transform(df_input)

    prob1 = model.predict_proba(df_scaled)[0][1]


    swapped = {}

    for k, v in fighter_row_dict.items():
        if k.startswith("red"):
            swapped["blue" + k[3:]] = v
        elif k.startswith("blue"):
            swapped["red" + k[4:]] = v
        elif k.startswith("diff"):
            swapped[k] = -v   
        else:
            swapped[k] = v

    df_swap = pd.DataFrame([swapped])

    for col in features:
        if col not in df_swap:
            df_swap[col] = 0

    df_swap = df_swap[features]
    df_swap_scaled = scaler.transform(df_swap)

    prob2 = model.predict_proba(df_swap_scaled)[0][1]


    prob = (prob1 + (1 - prob2)) / 2
    pred = int(prob >= 0.5)

    return {
        "predictedWinner": "Red" if pred == 1 else "Blue",
        "redWinProbability": float(prob),
        "confidence": float(max(prob, 1 - prob)),
        "topFeatures": importance.head(10).to_dict(orient="records")
    }


def build_fight_input(red, blue, is_title_fight, fight_length, is_womens_fight, selected_weight_class):
    fight = {}


    fight["maxRounds"] = 5 if fight_length == 25 else 3
    fight["isTitleFight"] = 1 if is_title_fight else 0
    fight["isWomensFight"] = 1 if is_womens_fight else 0
    fight["weightClass"] = weight_class_map[selected_weight_class]


    fight["redHeightCm"] = red["heightCm"]
    fight["redReachCm"] = red["reachCm"]
    fight["redStance"] = red["stanceEncoded"]
    fight["redAge"] = red["age"]

    fight["redKdPerMin"] = red["kdPerMin"]
    fight["redSigLandedPerMin"] = red["sigLandedPerMin"]
    fight["redSigThrownPerMin"] = red["sigThrownPerMin"]
    fight["redSigAccuracy"] = red["sigAccuracy"]
    fight["redSigAbsorbedPerMin"] = red["sigAbsorbedPerMin"]
    fight["redSigDefense"] = red["sigDefense"]

    fight["redTdPerMin"] = red["tdPerMin"]
    fight["redTdAccuracy"] = red["tdAccuracy"]
    fight["redTdDefense"] = red["tdDefense"]

    fight["redSubPerMin"] = red["subPerMin"]
    fight["redRevPerMin"] = red["revPerMin"]
    fight["redControlRatio"] = red["controlRatio"]

    fight["redWinRate"] = red["winRate"]
    fight["redRecentWinRate"] = red["recentWinRate"]
    fight["redTotalFights"] = red["totalFights"]
    fight["redDaysSinceLastFight"] = red["daysSinceLastFight"]
    fight["redFinishRate"] = red["finishRate"]

    fight["redRecentSigPerMin"] = red["recentSigPerMin"]
    fight["redRecentKdPerMin"] = red["recentKdPerMin"]
    fight["redRecentTdPerMin"] = red["recentTdPerMin"]

    fight["redCareerMinutes"] = red["careerMinutes"]
    fight["redAvgFightTime"] = red["avgFightTime"]
    fight["redWinStreak"] = red["winStreak"]

    fight["redActivityScore"] = red["activityScore"]
    fight["redIsRingRust"] = red["isRingRust"]
    fight["redPaceXAcc"] = red["paceXAcc"]
    fight["redKOThreat"] = red["koThreat"]


    fight["blueHeightCm"] = blue["heightCm"]
    fight["blueReachCm"] = blue["reachCm"]
    fight["blueStance"] = blue["stanceEncoded"]
    fight["blueAge"] = blue["age"]

    fight["blueKdPerMin"] = blue["kdPerMin"]
    fight["blueSigLandedPerMin"] = blue["sigLandedPerMin"]
    fight["blueSigThrownPerMin"] = blue["sigThrownPerMin"]
    fight["blueSigAccuracy"] = blue["sigAccuracy"]
    fight["blueSigAbsorbedPerMin"] = blue["sigAbsorbedPerMin"]
    fight["blueSigDefense"] = blue["sigDefense"]

    fight["blueTdPerMin"] = blue["tdPerMin"]
    fight["blueTdAccuracy"] = blue["tdAccuracy"]
    fight["blueTdDefense"] = blue["tdDefense"]

    fight["blueSubPerMin"] = blue["subPerMin"]
    fight["blueRevPerMin"] = blue["revPerMin"]
    fight["blueControlRatio"] = blue["controlRatio"]

    fight["blueWinRate"] = blue["winRate"]
    fight["blueRecentWinRate"] = blue["recentWinRate"]
    fight["blueTotalFights"] = blue["totalFights"]
    fight["blueDaysSinceLastFight"] = blue["daysSinceLastFight"]
    fight["blueFinishRate"] = blue["finishRate"]

    fight["blueRecentSigPerMin"] = blue["recentSigPerMin"]
    fight["blueRecentKdPerMin"] = blue["recentKdPerMin"]
    fight["blueRecentTdPerMin"] = blue["recentTdPerMin"]

    fight["blueCareerMinutes"] = blue["careerMinutes"]
    fight["blueAvgFightTime"] = blue["avgFightTime"]
    fight["blueWinStreak"] = blue["winStreak"]

    fight["blueActivityScore"] = blue["activityScore"]
    fight["blueIsRingRust"] = blue["isRingRust"]
    fight["bluePaceXAcc"] = blue["paceXAcc"]
    fight["blueKOThreat"] = blue["koThreat"]


    diff_pairs = [
        "HeightCm", "ReachCm", "Age",
        "KdPerMin", "SigLandedPerMin", "SigThrownPerMin", "SigAccuracy",
        "SigAbsorbedPerMin", "SigDefense",
        "TdPerMin", "TdAccuracy", "TdDefense",
        "SubPerMin", "RevPerMin", "ControlRatio",
        "WinRate", "RecentWinRate", "TotalFights", "DaysSinceLastFight",
        "FinishRate", "RecentSigPerMin", "RecentKdPerMin", "RecentTdPerMin",
        "CareerMinutes", "AvgFightTime", "WinStreak",
        "ActivityScore", "IsRingRust", "PaceXAcc", "KOThreat"
    ]

    for key in diff_pairs:
        fight["diff" + key] = fight["red" + key] - fight["blue" + key]


    eps = 1e-5

    fight["diffAgeSquared"] = fight["redAge"]**2 - fight["blueAge"]**2

    fight["sigLandedRatio"] = fight["redSigLandedPerMin"] / (fight["blueSigLandedPerMin"] + eps)
    fight["tdRatio"] = fight["redTdPerMin"] / (fight["blueTdPerMin"] + eps)

    fight["redEffectiveStriking"] = fight["redSigLandedPerMin"] - fight["blueSigAbsorbedPerMin"]
    fight["blueEffectiveStriking"] = fight["blueSigLandedPerMin"] - fight["redSigAbsorbedPerMin"]
    fight["diffEffectiveStriking"] = fight["redEffectiveStriking"] - fight["blueEffectiveStriking"]

    fight["redGrapplingEdge"] = fight["redTdAccuracy"] * (1 - fight["blueTdDefense"])
    fight["blueGrapplingEdge"] = fight["blueTdAccuracy"] * (1 - fight["redTdDefense"])
    fight["diffGrapplingEdge"] = fight["redGrapplingEdge"] - fight["blueGrapplingEdge"]

    fight["redStrikerVsGrappler"] = fight["redSigLandedPerMin"] * fight["blueTdPerMin"]
    fight["blueStrikerVsGrappler"] = fight["blueSigLandedPerMin"] * fight["redTdPerMin"]
    fight["diffStrikerVsGrappler"] = fight["redStrikerVsGrappler"] - fight["blueStrikerVsGrappler"]

    fight["redSubThreat"] = fight["redSubPerMin"] * fight["redTdPerMin"]
    fight["blueSubThreat"] = fight["blueSubPerMin"] * fight["blueTdPerMin"]
    fight["diffSubThreat"] = fight["redSubThreat"] - fight["blueSubThreat"]

    return fight


if "screen" not in st.session_state:
    st.session_state.screen = "input"


if st.session_state.screen == "input":

    st.title("UFC Fight Predictor")

    st.write("Fight Settings")

    is_womens_fight = st.toggle("Women's Fight", value=False)

    if is_womens_fight:
        allowed_weight_classes = [
            "Strawweight",
            "Flyweight",
            "Bantamweight"
        ]
    else:
        allowed_weight_classes = [
            "Flyweight",
            "Bantamweight",
            "Featherweight",
            "Lightweight",
            "Welterweight",
            "Middleweight",
            "Light Heavyweight",
            "Heavyweight"
        ]

    default_index = allowed_weight_classes.index("Lightweight") if "Lightweight" in allowed_weight_classes else 0

    selected_weight_class = st.selectbox(
        "Weight Class",
        allowed_weight_classes,
        index=default_index
    )

    is_title_fight = st.toggle("Championship Fight", value=False)

    if is_title_fight:
        fight_length = 25
        st.radio(
            "Fight Length",
            options=[25],
            format_func=lambda x: f"{x} minutes (5 rounds)",
            disabled=True,
            key="fight_length_locked"
        )
    else:
        fight_length = st.radio(
            "Fight Length",
            options=[15, 25],
            format_func=lambda x: f"{x} minutes ({3 if x == 15 else 5} rounds)",
            key="fight_length"
        )

    col1, col2 = st.columns(2)

    gender_value = 1 if is_womens_fight else 0

    filtered_fighters_df = fighters_df[
        fighters_df["latestIsWomensFight"] == gender_value
    ].copy()

    fighter_names = sorted(filtered_fighters_df["fighterName"].tolist())


    with col1:
        st.subheader("Red Fighter")
        red_name = st.selectbox("Select Fighter", fighter_names, key="red")
        red = filtered_fighters_df[filtered_fighters_df["fighterName"] == red_name].iloc[0]

        
        st.markdown(f"### {red_name}")

        st.write("Height:", red["heightCm"], "cm")
        st.write("Reach:", red["reachCm"], "cm")
        st.write("Sig Strikes/min:", round(red["sigLandedPerMin"], 2))
        st.write("Takedowns/min:", round(red["tdPerMin"], 2))
        st.write("Win Rate:", f"{red['winRate']:.1%}")
        st.write("Weight Class:", list(weight_class_map.keys())[int(red["latestWeightClass"])])


    with col2:
        st.subheader("Blue Fighter")
        blue_name = st.selectbox("Select Fighter", fighter_names, key="blue")
        blue = filtered_fighters_df[filtered_fighters_df["fighterName"] == blue_name].iloc[0]

        
        st.markdown(f"### {blue_name}")

        st.write("Height:", blue["heightCm"], "cm")
        st.write("Reach:", blue["reachCm"], "cm")
        st.write("Sig Strikes/min:", round(blue["sigLandedPerMin"], 2))
        st.write("Takedowns/min:", round(blue["tdPerMin"], 2))
        st.write("Win Rate:", f"{blue['winRate']:.1%}")
        st.write("Weight Class:", list(weight_class_map.keys())[int(blue["latestWeightClass"])])


    if st.button("Calculate Fight"):


        fight_input = build_fight_input(
            red,
            blue,
            is_title_fight,
            fight_length,
            is_womens_fight,
            selected_weight_class
        )
        result = predict_fight(fight_input)

        st.session_state.result = result
        st.session_state.red_name = red_name
        st.session_state.blue_name = blue_name

        st.session_state.screen = "result"
        st.rerun()


elif st.session_state.screen == "result":


    result = st.session_state.result
    red_name = st.session_state.red_name
    blue_name = st.session_state.blue_name

    red_prob = result["redWinProbability"]
    blue_prob = 1 - red_prob

    winner_name = red_name if red_prob > blue_prob else blue_name

    st.title("Fight Result")

    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown(f"### {red_name}")

    with col2:
        
        st.markdown(f"### {blue_name}")

    


    st.write("### Win Probability")
    st.progress(int(red_prob * 100))

    colA, colB = st.columns(2)
    colA.write(f" {red_name}: {red_prob:.2%}")
    colB.write(f" {blue_name}: {blue_prob:.2%}")


    st.subheader(f"Winner: {winner_name}")


    st.write("### Key Factors")

    feat_df = pd.DataFrame(result["topFeatures"]).copy()
    feat_df = feat_df.sort_values(by="importance", ascending=True)


    fig, ax = plt.subplots(figsize=(10, 6))

  
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")


    ax.barh(feat_df["feature"], feat_df["importance"])

    text_colour = "#E0E0E0"

    ax.tick_params(colors=text_colour)
    ax.xaxis.label.set_color(text_colour)
    ax.yaxis.label.set_color(text_colour)
    ax.title.set_color(text_colour)




    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")

    plt.tight_layout()

    st.pyplot(fig)



    if st.button("Reset"):
        st.session_state.screen = "input"
        st.rerun()