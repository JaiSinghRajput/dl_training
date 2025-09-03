import numpy as np
import pandas as pd

def generate_student_scores(n=1000, random_state=None, save_path=None):
    rng = np.random.default_rng(random_state)
    hours = rng.uniform(0.5, 9.5, size=n)
    hours = np.round(hours, 1)
    noise = rng.normal(loc=0.0, scale=8.0, size=n)
    scores = 10 * hours + noise
    scores = np.clip(scores, 0, 100)
    scores = np.round(scores, 0).astype(int)
    df = pd.DataFrame({"Hours": hours, "Scores": scores})
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"✅ Dataset saved to {save_path}")
    return df

df = generate_student_scores(save_path="student_scores.csv")
print(df.head())