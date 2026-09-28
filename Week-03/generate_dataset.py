import numpy as np
import pandas as pd

# ==================================================
# Generate Academic Performance Dataset
# ==================================================

np.random.seed(42)

# Number of students
n = 5000


# ==================================================
# 1. Generate Study Hours
# ==================================================
# Students study between approximately 1 and 15
# hours per week.

study_hours = np.clip(
    np.random.normal(
        loc=7,
        scale=3,
        size=n
    ),
    1,
    15
)


# ==================================================
# 2. Generate Attendance Percentage
# ==================================================
# Attendance ranges from 40% to 100%.

attendance = np.clip(
    np.random.normal(
        loc=75,
        scale=12,
        size=n
    ),
    40,
    100
)


# ==================================================
# 3. Calculate Academic Performance Score
# ==================================================
#
# Study hours and attendance are the ONLY factors
# used to generate the result.
#
# Study hours contribute 55%.
# Attendance contributes 45%.
#
# A small random component is added to represent
# natural variation between students.
# ==================================================

study_score = np.minimum(
    study_hours / 10,
    1
)

attendance_score = np.minimum(
    attendance / 80,
    1
)

noise = np.random.normal(
    loc=0,
    scale=0.05,
    size=n
)

performance_score = (
    0.55 * study_score
    + 0.45 * attendance_score
    + noise
)


# ==================================================
# 4. Convert Performance Score to Pass / Fail
# ==================================================
#
# 1 = Pass
# 0 = Fail
#
# A score of 0.70 or above is considered Pass.
# ==================================================

result = (
    performance_score >= 0.70
).astype(int)


# ==================================================
# 5. Create DataFrame
# ==================================================

df = pd.DataFrame({

    "study_hours":
        np.round(study_hours, 2),

    "attendance":
        np.round(attendance, 2),

    "result":
        result
})


# ==================================================
# 6. Shuffle Dataset
# ==================================================

df = df.sample(

    frac=1,

    random_state=42

).reset_index(drop=True)


# ==================================================
# 7. Save Dataset
# ==================================================

df.to_csv(

    "student_performance.csv",

    index=False
)


# ==================================================
# 8. Display Dataset Information
# ==================================================

print("=" * 55)
print("STUDENT DATASET GENERATED SUCCESSFULLY")
print("=" * 55)

print(
    f"\nNumber of students: {len(df)}"
)

print("\nFirst 10 records:")

print(
    df.head(10)
)

print("\nClass distribution:")

print(
    df["result"].value_counts()
)

print("\nClass percentages:")

print(
    (
        df["result"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nDataset statistics:")

print(
    df.describe()
)

print(
    "\nSaved as: student_performance.csv"
)