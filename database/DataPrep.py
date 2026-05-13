import pandas as pd

#Class for Loading and cleaning the data
class DataPreparator:
    def __init__(self):
        pass
    def read_data(self, path):
        df = pd.read_csv(path)

        return self.clean_data(df)


    def clean_data(self, df):
        df = df[["book_name", "summaries"]]
        df = df.dropna()
        df = df.drop_duplicates(subset=["book_name"])

        return df.to_dict("records")
        