import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

"""
STREAMLIT DASHBOARD FOR FANTASY SQUAD
------------------------------------

Features:
- Field visualization by position
- Player info panel (age, nationality, team, fatigue, predicted score)
- Interactive filters
"""


# -------------------------
# Load team data
# -------------------------
@st.cache_data
def load_team(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


# -------------------------
# Plot football field
# -------------------------
def plot_field(df):
    # Basic football field dimensions
    field_length = 100
    field_width = 60

    fig = go.Figure()

    # Draw field rectangle
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=field_length,
        y1=field_width,
        line=dict(color="green"),
        fillcolor="green",
        layer="below",
    )

    # Assign positions on the field
    positions_x = {"GK": 10, "DEF": 30, "MID": 60, "FWD": 85}

    positions_y = {
        "GK": [30],
        "DEF": [10, 25, 40, 55],
        "MID": [10, 25, 40, 55],
        "FWD": [20, 40],
    }

    # Plot players
    for pos, x in positions_x.items():
        y_coords = positions_y.get(pos, [30])
        players = df[df["position"] == pos]
        for i, (_, player) in enumerate(players.iterrows()):
            y = y_coords[i % len(y_coords)]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=30, color="blue"),
                    text=player["name"],
                    textposition="bottom center",
                    hovertemplate=(
                        f"<b>{player['name']}</b><br>"
                        f"Position: {player['position']}<br>"
                        f"Team: {player['team']}<br>"
                        f"Age: {player.get('age','N/A')}<br>"
                        f"Nationality: {player.get('nationality','N/A')}<br>"
                        f"Predicted Score: {player.get('predicted_score','N/A')}<br>"
                        f"Fatigue: {player.get('fatigue','N/A')}"
                    ),
                )
            )

    fig.update_layout(
        xaxis=dict(
            range=[0, field_length], showgrid=False, zeroline=False, visible=False
        ),
        yaxis=dict(
            range=[0, field_width], showgrid=False, zeroline=False, visible=False
        ),
        height=600,
        width=900,
        plot_bgcolor="green",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# -------------------------
# Streamlit app
# -------------------------
st.title("⚽ Fantasy Squad Optimizer Dashboard")

# Load team
team_file = st.sidebar.file_uploader("Upload Optimized Team JSON", type=["json"])
if team_file is not None:
    df_team = load_team(team_file)

    # Optional filters
    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Minimum predicted score", 0.0, 100.0, 0.0)
    df_team_filtered = df_team[df_team["predicted_score"] >= min_score]

    st.subheader("Team Table")
    st.dataframe(
        df_team_filtered[
            [
                "name",
                "position",
                "team",
                "age",
                "nationality",
                "predicted_score",
                "fatigue",
            ]
        ]
    )

    st.subheader("Field Visualization")
    fig = plot_field(df_team_filtered)
    st.plotly_chart(fig)
else:
    st.warning("Upload your optimized team JSON file to visualize the squad.")
