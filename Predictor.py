import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Advanced Cricket AI Predictor", layout="wide")
st.title("Cricket Score Prediction Tool (Innings 1 & 2)")

uploaded_file = st.file_uploader("Upload CSV Ball-by-Ball Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["runs_total"] = df["Runs"]
    df["is_wicket"] = df["Dismissal Type"].notna().astype(int)

    st.success("Data uploaded and processed!")

    innings_choice = st.selectbox("Select Innings", [1, 2])

    venues = sorted(df["Venue"].unique())
    venue = st.selectbox("Venue", venues)

    if innings_choice == 1:
        team1 = st.selectbox("Batting Team", sorted(df["Batting Team"].unique()))
        team2 = st.selectbox("Bowling Team", sorted(df["Bowling Team"].unique()))
        score = st.number_input("Current Score", min_value=0, value=50)
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
        overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

        # Stats
        venue_df = df[df["Venue"] == venue]
        venue_avg_score = venue_df.groupby(["Match_ID", "Innings"])["runs_total"].sum().mean()
        venue_wpb = venue_df["is_wicket"].sum() / len(venue_df)

        team_df = df[(df["batting_team"] == team1) & (df["venue"] == venue)]
        team_avg_score = team_df.groupby("match_id")["runs_total"].sum().mean()

        form = team_df.groupby("match_id")["runs_total"].sum().rolling(5).mean().mean()

        st.metric("Venue Avg Score", f"{venue_avg_score:.2f}")
        st.metric("Wickets per Ball (Venue)", f"{venue_wpb:.3f}")
        st.metric("Team Avg Score (Venue)", f"{team_avg_score:.1f}")
        st.metric("Team Recent Form (5 Match Avg)", f"{form:.1f}")

        # Feature Engineering
        df_model = df.groupby(["match_id", "inning", "venue", "batting_team", "bowling_team"]).agg({
            "runs_total": "sum",
            "is_wicket": "sum"
        }).reset_index()
        df_model["wickets_per_ball"] = df_model["is_wicket"] / 120

        X = df_model[["wickets_per_ball"]]
        y = df_model["runs_total"]

        model = RandomForestRegressor()
        model.fit(X, y)

        input_data = pd.DataFrame({
            "wickets_per_ball": [venue_wpb]
        })

        pred_score = model.predict(input_data)[0]

        st.subheader(f"Predicted Final Score: {int(pred_score)}")
        st.success(f"To Defend Well, Target Should Be ~ {int(pred_score + 10)}")

    elif innings_choice == 2:
        team1 = st.selectbox("1st Innings Batting Team", sorted(df["batting_team"].unique()))
        team2 = st.selectbox("1st Innings Bowling Team", sorted(df["bowling_team"].unique()))
        score = st.number_input("1st Innings Score", min_value=0, value=160)
        wickets = st.number_input("1st Innings Wickets", min_value=0, max_value=10, value=7)
        overs = st.number_input("1st Innings Overs", min_value=0.0, max_value=20.0, value=20.0, step=0.1)

        target = score + 1
        st.markdown(f"### Target for 2nd Innings: {target} runs")

        st.markdown("#### 2nd Innings Teams Auto-filled")
        st.write("Batting Team (2nd Innings):", team2)
        st.write("Bowling Team (2nd Innings):", team1)

        venue_df = df[df["venue"] == venue]
        wpb = venue_df["is_wicket"].sum() / len(venue_df)

        model = RandomForestRegressor()
        df_model = df.groupby(["match_id", "inning", "venue", "batting_team", "bowling_team"]).agg({
            "runs_total": "sum",
            "is_wicket": "sum"
        }).reset_index()
        df_model["wickets_per_ball"] = df_model["is_wicket"] / 120
        X = df_model[["wickets_per_ball"]]
        y = df_model["runs_total"]
        model.fit(X, y)

        pred_score = model.predict([[wpb]])[0]

        st.subheader(f"Predicted Score in 2nd Innings: {int(pred_score)}")

        if pred_score >= target:
            overs_taken = round((target / pred_score) * 20, 2)
            st.success(f"Target may be chased in approximately {overs_taken} overs.")
        else:
            st.error("Target may not be chased based on prediction.")

        st.markdown("### Over-wise Target Table")
        over_table = []
        for o in range(1, 21):
            run_required = round((o / 20) * target)
            over_table.append({"Over": o, "Cumulative Target": run_required})
        st.table(pd.DataFrame(over_table))

else:
    st.info("Please upload your ball-by-ball CSV data to get started.")
