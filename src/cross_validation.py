import pandas as pd
from sklearn import model_selection

"""
- binary classification
- multi class classification
- multi label classification
- single column regression
- multi column regression
- holdout
"""

class CrossValidation:
    def __init__(
        self,
        df,
        target_cols,
        shuffle,    
        problem_type="binary_classification",
        multilabel_delimiter=",",
        num_folds=5,
        random_state=42,
        ):

        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.num_folds = num_folds
        # set "shuffle=False" for Time_Series Data
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, 
                                                                     y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multicol_regression"):
            if self.problem_type == "single_col_regression" and self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            elif self.problem_type == "multicol_regression" and self.num_targets < 2:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds, shuffle=False)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
                
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'kfold'] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, 'kfold'] = 1
            
        elif self.problem_type == "multilabel_classifier":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
             
        else:
            raise Exception("Problem type not understood!")

        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    cv = CrossValidation(df, target_cols=["target"], 
                         shuffle=True, problem_type="holdout_10")
    split_df = cv.split()
    print(split_df["kfold"].value_counts())
    