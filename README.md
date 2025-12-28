F1 Pit Boss: The 24-Hour ML Challenge

    "Races are won on the track, but championships are won in the data."

About The Project

I set myself a challenge: Build a useful Machine Learning project in exactly 24 hours.

I chose Formula 1 because the margin for error is measured in milliseconds. Teams pay millions for proprietary strategy software, so I wanted to see if I could build a "Lite" version of a Race Strategy Engineer from scratch using open-source tools.

This tool analyzes historical race data (2018-2024) to predict optimal pit windows. It transforms raw lap times into actionable strategy insights, helping predict exactly when a driver must pit before the "cliff" in tire performance destroys their race.

What's Under the Hood?
The Data Challenge: Contextual Feature Engineering

The core difficulty wasn't just loading data—it was teaching the model "race context." Raw F1 data gives you lap times, but it doesn't explicitly tell you how old the tires are or why a lap was slow (was it traffic? a yellow flag? or just old rubber?).

To fix this, I wrote custom logic to:

    Reconstruct "Stints": Algorithmically detecting pit stops (outliers in lap time) to segment the race into distinct stints.

    Calculate Tire Age: A counter that resets dynamically every time a driver hits the pit lane.

    Filter the Noise: Processing 20,000+ laps to isolate pure racing laps, removing Safety Car eras and VSC (Virtual Safety Car) periods that would skew the training data.

The Engine: XGBoost

I used XGBoost as the core regressor because of its ability to handle non-linear relationships in tabular data better than standard linear regression.

The "Track Length" Problem: One major hurdle was normalizing performance across different circuits.

    A "fast" lap at Spa-Francorchamps is ~105 seconds.

    A "fast" lap at Monaco is ~75 seconds.

If you feed this data blindly, the model gets confused by the massive variance in base lap times. However, the XGBoost model successfully learned to "recognize" the difference between circuits. It understands that a 1:20 lap is competitive in Belgium but disastrous in Austria, allowing it to predict tire degradation trends accurately regardless of the specific Grand Prix location.
Tech Stack

    Python 3.10: The core logic.

    Streamlit: For the interactive strategy dashboard.

    Pandas: For heavy-duty data merging and time-series logic.

    XGBoost: The gradient boosting framework used for the prediction model.

    Seaborn/Matplotlib: For visualizing the degradation curves.

    Kaggle API: Source of the F1 World Championship dataset (2018-2024).

How to Run It

    Clone the repo:
    Bash

git clone https://github.com/vichruth/pit-window-predictor.git
cd pit-window-predictor

Install dependencies:
Bash

pip install -r requirements.txt

Launch the Dashboard:
Bash

    cd codes
    streamlit run app.py

Results

The model successfully identifies the "crossover point"—where the time lost due to old tires exceeds the time cost of a pit stop. (See the screenshot above: The red line represents the AI's predicted pace strategy vs. the actual race data in grey).The output of how the model looks is stored as image in output folder.
