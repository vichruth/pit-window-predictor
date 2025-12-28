import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    df = pd.read_csv('../f1_dataset/f1_feature_engineered_data.csv')
    return df

@st.cache_resource
def load_model():
    with open('../outputs/pit_boss_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

st.title("F1 Pit Boss")

# Load data
try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Error: Could not find dataset or model. Make sure you are running this from the 'codes' folder!")
    st.stop()


st.sidebar.header("Race Settings")
years = sorted(df['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Season", years)
races_in_year = df[df['year'] == selected_year]['name'].unique()
selected_race = st.sidebar.selectbox("Select Grand Prix", races_in_year)
race_id = df[(df['year'] == selected_year) & (df['name'] == selected_race)]['raceId'].iloc[0]
drivers_in_race = df[df['raceId'] == race_id]['code'].unique()
selected_driver = st.sidebar.selectbox("Select Driver", drivers_in_race)

#Filter Data
race_data = df[
    (df['raceId'] == race_id) & 
    (df['code'] == selected_driver)
].copy()

#Predict
if not race_data.empty:
    features = ['circuitId', 'driverId', 'tire_age', 'lap']
    X_input = race_data[features]
    race_data['predicted_time_ms'] = model.predict(X_input)
    race_data['Actual Time (s)'] = race_data['milliseconds'] / 1000
    race_data['AI Predicted (s)'] = race_data['predicted_time_ms'] / 1000
    st.subheader(f"Strategy Analysis: {selected_driver} @ {selected_race}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=race_data, x='lap', y='Actual Time (s)', 
                 label='Actual Pace', color='grey', alpha=0.5, ax=ax)
    sns.lineplot(data=race_data, x='lap', y='AI Predicted (s)', 
                 label='AI Strategy Model', color='red', linewidth=2, ax=ax)
    pit_stops = race_data[race_data['is_pit'] == True]
    if not pit_stops.empty:
        plt.scatter(pit_stops['lap'], pit_stops['Actual Time (s)'], 
                    color='gold', s=100, edgecolors='black', label='Pit Stop', zorder=5)
    ax.set_ylabel("Lap Time (s)")
    ax.set_xlabel("Lap Number")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    #Stats
    avg_error = (race_data['Actual Time (s)'] - race_data['AI Predicted (s)']).abs().mean()
    st.metric("Model Accuracy (This Race)", f"±{avg_error:.3f} s")

else:
    st.warning("No data found for this specific driver/race combo.")
    