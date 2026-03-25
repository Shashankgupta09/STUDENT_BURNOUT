import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

data = pd.DataFrame({
    "study_hours": np.random.randint(2, 13, rows),
    "sleep_hours": np.random.randint(3, 12, rows),
    "screen_time": np.random.randint(2, 18, rows),
    "exercise_minutes": np.random.randint(0, 90, rows),
    "assignment_load": np.random.randint(1, 15, rows),
    "anxiety_level": np.random.randint(0, 4, rows)
})

def burnout_logic(row):
    score = (
        row["study_hours"] * 0.3 +
        row["screen_time"] * 0.3 +
        row["assignment_load"] * 0.4 +
        row["anxiety_level"] * 2 -
        row["sleep_hours"] * 0.5 -
        row["exercise_minutes"] * 0.02
    )
    
    if score < 5:
        return 0
    elif score < 10:
        return 1
    else:
        return 2

data["burnout"] = data.apply(burnout_logic, axis=1)

data.to_csv("burnout_dataset.csv", index=False)

print("Dataset created with 500 rows")
