import joblib
import pandas as pd
import os
from text_cleaning_module import TextCleaner

# Load the text cleaner
cleaner = TextCleaner()

# Get the base directory relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Load the two models
case_sought_model = joblib.load("models/best_case_action_sought.joblib")
case_group_model = joblib.load("models/best_case_type_model.joblib")

# Label maps (must match training time!)
sought_label_map = {0: "No", 1: "Yes"}

def predict_case_attributes(full_case):
    """
    Takes raw full_case input (as a list of paragraphs or single string),
    cleans it, and returns predictions from both models.
    """
    # Wrap in Series for compatibility
    case_series = pd.Series([full_case])

    # Apply cleaning
    cleaned = cleaner.transform(case_series)

    # Predict
    sought_pred = case_sought_model.predict(cleaned)[0]
    group_pred = case_group_model.predict(cleaned)[0]

    return {
        "Class Action Sought": sought_label_map[sought_pred],
        "Predicted Case Group": group_pred
    }

# Example usage
if __name__ == "__main__":
    example_case = [
        "This action is brought on behalf of a class of workers...",
        "The defendant has failed to provide compensation..."
    ]
    results = predict_case_attributes(example_case)
    print("Predictions:")
    for k, v in results.items():
        print(f"{k}: {v}")
