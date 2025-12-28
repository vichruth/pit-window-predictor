F1 Pit Boss: The 24-Hour ML Challenge

    "Races are won on the track, but championships are won in the data."
About The Project

I set myself a challenge: Build a useful Machine Learning project in exactly 24 hours. I chose Formula 1 because the margin for error is milliseconds. Teams pay millions for strategy software, so I wanted to see if I could build a "Lite" version of a Race Strategy Engineer from scratch.

This tool analyzes historical race data (2018-2024) to calculate Tire Degradation. It transforms raw lap times into actionable strategy insights by predicting when a driver must pit before they lose too much time.

What's Under the Hood?

The core challenge wasn't just "loading data"—it was Contextual Feature Engineering. Raw F1 data gives you lap times, but it doesn't tell you how old the tires are. I wrote custom logic to:

    Reconstruct "Stints": Algorithmically detecting pit stops to segment the race.

    Calculate Tire Age: A counter that resets dynamically every time a driver hits the pit lane.

    Filter the Noise: Processing 20,000+ laps to isolate pure racing laps from Safety Car eras.

Tech Stack

    Python 3.10 (The engine)

    Pandas (For heavy-duty data merging and time-series logic)

    Seaborn/Matplotlib (For visualizing the degradation curves)

    Kaggle API (Source of the F1 World Championship dataset)
