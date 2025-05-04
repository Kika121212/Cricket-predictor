import streamlit as st 
import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Advanced Cricket AI Predictor", layout="wide") st.title("Cricket Score Prediction Tool (Innings 1 & 2) - Advanced ML")

uploaded_file = st.file_uploader("Upload CSV Ball-by-Ball Data", type="csv")

if uploaded_file: df = pd.read_csv(uploaded_file) df["runs_total"] = df["Runs"] df["is_wicket"] = df["Dismissal Type"].notna().astype(int)

st.success("Data uploaded and processed!")

innings_choice = st.selectbox("Select Innings", [1, 2])
venues = sorted(df["Venue"].dropna().unique())
venue = st.selectbox("Venue", venues)

overall_venue_stats = df[df["Venue"] == venue].groupby(["Match ID", "Innings"]).agg({
    "runs_total": "sum",
    "is_wicket": "sum"
}).reset_index()
venue_avg_score = overall_venue_stats["runs_total"].mean()
venue_wpb = overall_venue_stats["is_wicket"].sum() / (len(overall_venue_stats) * 120)

st.metric("Overall Venue Avg Score", f"{venue_avg_score:.1f}")
st.metric("Wickets per Ball (Venue)", f"{venue_wpb:.3f}")

if innings_choice == 1:
    team1 = st.selectbox("Batting Team", sorted(df["Batting Team"].unique()))
    team2 = st.selectbox("Bowling Team", sorted(df["Bowling Team"].unique()))
    score = st.number_input("Current Score", min_value=0, value=50)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

    team_form_df = df[(df["Batting Team"] == team1)].groupby("Match ID")["runs_total"].sum().rolling(5).mean().mean()
    run_rate = score / overs if overs > 0 else 0

    st.metric("Team Recent Form (5 Match Avg)", f"{team_form_df:.1f}")
    st.metric("Current Run Rate", f"{run_rate:.2f}")

    # ML Model
    df_model = overall_venue_stats.copy()
    df_model["wickets_per_ball"] = df_model["is_wicket"] / 120
    X = df_model[["wickets_per_ball"]]
    y = df_model["runs_total"]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    input_data = pd.DataFrame({
        "wickets_per_ball": [venue_wpb]
    })

    pred_score = model.predict(input_data)[0]
    st.subheader(f"Predicted Final Score: {int(pred_score)}")
    st.success(f"Target for 2nd Innings should be ~ {int(pred_score + 10)}")

else:
    team1 = st.selectbox("1st Innings Batting Team", sorted(df["Batting Team"].unique()))
    team2 = st.selectbox("1st Innings Bowling Team", sorted(df["Bowling Team"].unique()))
    score = st.number_input("1st Innings Score", min_value=0, value=160)
    wickets = st.number_input("1st Innings Wickets", min_value=0, max_value=10, value=7)
    overs = st.number_input("1st Innings Overs", min_value=0.0, max_value=20.0, value=20.0, step=0.1)

    target = score + 1
    st.markdown(f"### Target for 2nd Innings: {target} runs")
    st.markdown("#### 2nd Innings Teams Auto-filled")
    st.write("Batting Team (2nd Innings):", team2)
    st.write("Bowling Team (2nd Innings):", team1)

    df_model = overall_venue_stats.copy()
    df_model["wickets_per_ball"] = df_model["is_wicket"] / 120
    X = df_model[["wickets_per_ball"]]
    y = df_model["runs_total"]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    pred_score = model.predict([[venue_wpb]])[0]
    st.subheader(f"Predicted 2nd Innings Score: {int(pred_score)}")

    if pred_score >= target:
        overs_taken = round((target / pred_score) * 20, 2)
        st.success(f"Likely to chase in ~{overs_taken} overs")
    else:
        st.error("Target may not be chased based on prediction")

    # Dynamic Over-Wise Projection using predicted score
    st.markdown("### Over-wise Target Table (Projected to Chase)")
    over_table = []
    for o in range(1, 21):
        proj_score = round((o / 20) * pred_score)
        over_table.append({"Over": o, "Projected Score": proj_score})

    st.table(pd.DataFrame(over_table))

else: st.info("Please upload your ball-by-ball CSV data to get started.")
