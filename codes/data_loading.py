import pandas as pd
import os 
def load_and_merge_data():
    dataset_path = "../f1_dataset"
    print("Loading and merging datasets from:",dataset_path)
    # load races data
    races= pd.read_csv(os.path.join(dataset_path,'races.csv'))
    races = races[races['year']>=2018]
    races= races[['raceId', 'year', 'round', 'circuitId', 'name', 'date']]
    lap_times = pd.read_csv(os.path.join(dataset_path, 'lap_times.csv'))
    merged_df = pd.merge(lap_times, races, on='raceId', how='inner')

    #Load Drivers (For names)
    drivers = pd.read_csv(os.path.join(dataset_path, 'drivers.csv'))
    drivers = drivers[['driverId', 'driverRef', 'code']]
    merged_df = pd.merge(merged_df, drivers, on='driverId', how='left')

    #Load Pit Stops
    pit_stops = pd.read_csv(os.path.join(dataset_path, 'pit_stops.csv'))
    pit_stops = pit_stops.rename(columns={'stop': 'pit_stop_number', 'lap': 'pit_lap', 'time': 'pit_duration'})
    pit_stops = pit_stops[['raceId', 'driverId', 'pit_lap', 'pit_stop_number', 'pit_duration']]
    
    # Merge logic
    merged_df = pd.merge(merged_df, pit_stops,left_on=['raceId', 'driverId', 'lap'],right_on=['raceId', 'driverId', 'pit_lap'], how='left')
    merged_df['pit_stop_number'] = merged_df['pit_stop_number'].fillna(0)

    print(f"Data Loaded Successfully! Total Rows: {len(merged_df)}")
    return merged_df

if __name__ == "__main__":
    df = load_and_merge_data()
    df.to_csv('../f1_dataset/processed_laps_2018_2024.csv', index=False)
    print("Saved processed data to ../f1_dataset/processed_laps_2018_2024.csv")
    
    
    