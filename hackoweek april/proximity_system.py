import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime

# -----------------------------------
# 1. LOAD DATA + CLEANING
# -----------------------------------
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    print("Initial Data:")
    print(df.head())

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Remove outliers using IQR
    if 'distance' in df.columns:
        Q1 = df['distance'].quantile(0.25)
        Q3 = df['distance'].quantile(0.75)
        IQR = Q3 - Q1

        df = df[(df['distance'] >= Q1 - 1.5 * IQR) &
                (df['distance'] <= Q3 + 1.5 * IQR)]

    print("\nCleaned Data:")
    print(df.head())

    return df


# -----------------------------------
# 2. AGV GEOFENCING SYSTEM
# -----------------------------------
SAFE_DISTANCE = 2.0  # meters

def agv_control(distance):
    if distance < SAFE_DISTANCE:
        print("⚠ Worker detected! Slowing down AGV")
        log_incident(distance)
        return "SLOW"
    else:
        return "NORMAL"

def log_incident(distance):
    with open("sd_card_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - ALERT! Distance: {distance}\n")


# -----------------------------------
# 3. TIME SERIES ANALYSIS
# -----------------------------------
def plot_time_series(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['distance'])
        plt.title("Proximity Distance Over Time")
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    avg_distance = df['distance'].mean()
    print("\nAverage Distance:", avg_distance)


# -----------------------------------
# 4. BLE RSSI CLASSIFICATION
# -----------------------------------
def classify_rssi(rssi):
    if rssi > -50:
        return "Strong (Very Close)"
    elif rssi > -70:
        return "Medium"
    else:
        return "Weak (Far)"

def process_rssi_data(rssi_values):
    results = [classify_rssi(r) for r in rssi_values]
    print("\nRSSI Classification:", results)
    return results


# -----------------------------------
# 5. SENSOR FAILURE ML MODEL
# -----------------------------------
def train_failure_model(df):
    if 'failure' not in df.columns:
        print("\nNo 'failure' column found. Skipping ML model.")
        return

    # Feature Engineering
    df['rolling_mean'] = df['distance'].rolling(5).mean()
    df['rolling_std'] = df['distance'].rolling(5).std()

    df.dropna(inplace=True)

    X = df[['distance', 'rolling_mean', 'rolling_std']]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print("\nML Model Accuracy:", accuracy)


# -----------------------------------
# MAIN EXECUTION
# -----------------------------------
if __name__ == "__main__":

    file_path = "proximity_data.csv"  # Replace with your Kaggle CSV file

    # Step 1: Load & clean
    df = load_and_clean_data(file_path)

    # Step 2: Simulate AGV control using first few readings
    print("\nAGV Simulation:")
    for d in df['distance'].head(10):
        status = agv_control(d)
        print(f"Distance: {d} → AGV: {status}")

    # Step 3: Time series analysis
    plot_time_series(df)

    # Step 4: BLE RSSI simulation
    rssi_values = [-45, -60, -75, -85]
    process_rssi_data(rssi_values)

    # Step 5: ML model
    train_failure_model(df)
