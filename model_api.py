import os
import pandas as pd
import numpy as np
import pickle

class DiseaseRecommender:
    def __init__(self):
        # Find absolute path to this file (backend/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # datasets are in backend/datasets/
        datasets_path = os.path.join(base_dir, "datasets")
        # Model is at root/models/svc.pkl
        model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "svc.pkl"))

        self.description = pd.read_csv(os.path.join(datasets_path, "description.csv"))
        self.diets = pd.read_csv(os.path.join(datasets_path, "diets.csv"))
        self.medications = pd.read_csv(os.path.join(datasets_path, "medications.csv"))
        self.precautions = pd.read_csv(os.path.join(datasets_path, "precautions_df.csv"))
        self.workout = pd.read_csv(os.path.join(datasets_path, "workout_df.csv"))

        # Safe load model
        with open(model_path, "rb") as f:
            self.svc = pickle.load(f)

        # For constructing input vector and mapping labels
        training = pd.read_csv(os.path.join(datasets_path, "Training.csv"))
        self.symptoms_dict = {name: idx for idx, name in enumerate(training.columns[:-1])}
        self.diseases_list = sorted(self.description["Disease"].unique())

    def predict(self, input_symptoms):
        x_input = np.zeros(len(self.symptoms_dict))
        for symptom in input_symptoms:
            idx = self.symptoms_dict.get(symptom)
            if idx is not None:
                x_input[idx] = 1
        pred = self.svc.predict([x_input])[0]
        if isinstance(pred, int) and pred < len(self.diseases_list):
            return self.diseases_list[pred]
        return pred

    def get_details(self, disease):
        # Defensive: .empty checks if disease not in sheet
        desc = " ".join(self.description[self.description["Disease"] == disease]["Description"].values)
        prec = []
        if not self.precautions[self.precautions["Disease"] == disease].empty:
            prec = (self.precautions[self.precautions["Disease"] == disease].iloc[0, 1:]).dropna().tolist()
        meds = self.medications[self.medications["Disease"] == disease]["Medication"].dropna().tolist()
        diets = self.diets[self.diets["Disease"] == disease]["Diet"].dropna().tolist()
        workouts = []
        if "disease" in self.workout.columns:
            workouts = self.workout[self.workout["disease"] == disease]["workout"].dropna().tolist()
        return {
            "description": desc,
            "precautions": prec,
            "medications": meds,
            "diets": diets,
            "workouts": workouts,
        }
