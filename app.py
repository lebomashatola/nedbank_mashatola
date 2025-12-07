# app.py
import streamlit as st
from lineup_generator import generate_lineups

st.set_page_config(page_title="Fantasy Football Lineup", layout="wide")
st.title("Fantasy Football Optimal Lineup Against Opponent")

# Sidebar for opponent features
st.sidebar.header("Opponent Features")
elo = st.sidebar.slider("Opponent Elo", 1000, 2000, 1450)
avg_goals = st.sidebar.slider("Opponent Avg Goals Conceded", 0.0, 5.0, 1.2)
home_adv = st.sidebar.checkbox("Home Advantage", True)

opponent_features = {
    "Elo_rating": elo,
    "avg_goals_conceded": avg_goals,
    "home_advantage": int(home_adv)
}

# Generate lineups
ml_lineup, ml_points, top_lineup, mean_random_points, top_points = generate_lineups(opponent_features)

# Display total points
st.subheader("Total Predicted Points")
st.write(f"Predicted Lineup Points: {ml_points:.2f}")
st.write(f"Random Baseline Points (mean): {mean_random_points:.2f}")
st.write(f"Top Historical Points Baseline: {top_points:.2f}")

# Function to display lineup by position
def display_lineup(df, title="Lineup"):
    st.subheader(title)
    positions = ["GK","DEF","MID","FWD"]
    for pos in positions:
        pos_df = df[df["position"]==pos].sort_values("predicted_points", ascending=False)
        if not pos_df.empty:
            st.write(f"**{pos}**")
            st.table(pos_df[["name","salary","predicted_points"]])

display_lineup(ml_lineup, title="ML + Opponent Optimal Lineup")
display_lineup(top_lineup, title="Top Historical Points Baseline")

# Bar chart
st.subheader("Lineup Predicted Points Comparison")
comparison = {
    "ML Lineup": ml_points,
    "Random Baseline (mean)": mean_random_points,
    "Top Historical": top_points
}
st.bar_chart(comparison)
