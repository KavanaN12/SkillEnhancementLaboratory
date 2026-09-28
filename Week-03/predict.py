import os

import numpy as np
import pandas as pd
import tensorflow as tf


# ==================================================
# 1. File paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "academic_performance_dnn.keras"
)

SCALER_MEAN_PATH = os.path.join(
    RESULTS_DIR,
    "scaler_mean.npy"
)

SCALER_SCALE_PATH = os.path.join(
    RESULTS_DIR,
    "scaler_scale.npy"
)


# ==================================================
# 2. Check whether model exists
# ==================================================

required_files = [

    MODEL_PATH,

    SCALER_MEAN_PATH,

    SCALER_SCALE_PATH
]


for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            "\nTrained model files were not found."
        )

        print(
            "Please run the following first:"
        )

        print(
            "\npython generate_dataset.py"
        )

        print(
            "python train.py"
        )

        raise SystemExit


# ==================================================
# 3. Load trained model
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==================================================
# 4. Load scaler information
# ==================================================

scaler_mean = np.load(
    SCALER_MEAN_PATH
)

scaler_scale = np.load(
    SCALER_SCALE_PATH
)


# ==================================================
# 5. Get student information
# ==================================================

print("=" * 50)
print("STUDENT PASS / FAIL PREDICTION")
print("=" * 50)

try:

    study_hours = float(
        input(
            "\nEnter study hours per week: "
        )
    )

    attendance = float(
        input(
            "Enter attendance percentage: "
        )
    )


    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------

    if study_hours < 0:

        raise ValueError(
            "Study hours cannot be negative."
        )


    if attendance < 0 or attendance > 100:

        raise ValueError(
            "Attendance must be between 0 and 100."
        )


    # --------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------
    #
    # Feature order MUST be:
    #
    # 1. study_hours
    # 2. attendance
    # --------------------------------------------------

    student = pd.DataFrame(

        [[
            study_hours,
            attendance
        ]],

        columns=[
            "study_hours",
            "attendance"
        ]
    )


    # --------------------------------------------------
    # Apply the same scaling used during training
    # --------------------------------------------------

    student_scaled = (

        student.values
        - scaler_mean

    ) / scaler_scale


    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    probability = float(

        model.predict(

            student_scaled,

            verbose=0

        )[0][0]

    )


    # --------------------------------------------------
    # Convert probability to Pass / Fail
    # --------------------------------------------------

    if probability >= 0.50:

        result = "Pass"

    else:

        result = "Fail"


    # ==================================================
    # 6. Display result
    # ==================================================

    print("\n")
    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print(
        f"\nStudy Hours      : {study_hours:.2f}"
    )

    print(
        f"Attendance       : {attendance:.2f}%"
    )

    print(
        f"Pass Probability : {probability:.2%}"
    )

    print(
        f"Predicted Result : {result}"
    )

    print("\n")


except ValueError as error:

    print(
        f"\nInvalid input: {error}"
    )