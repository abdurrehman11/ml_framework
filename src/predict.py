import os
import pandas as pd
import numpy as np
from sklearn import metrics
import joblib

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)

        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_.pkl"))
        for col in cols:
            lbl = encoders[col]
            df.loc[:, col] = lbl.transform(df[col].values.tolist())

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns = ["id", "target"])

    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
    
