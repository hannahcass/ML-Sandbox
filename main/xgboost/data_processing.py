
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreProcessing:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def main(self, threshold: float) -> pd.DataFrame:
        empty_values = self.check_empty_values()
        if empty_values > 0:
            self.data = self.drop_empty_if_more_than_threshold(threshold)
            self.data = self.fill_empty_values_with_0()
        return self.data

    def check_empty_values(self):
        sum_empty_values = self.data.isnull().sum().sum()
        return sum_empty_values

    def drop_empty_if_more_than_threshold(self, threshold):
        value = len(self.data.columns) * threshold
        self.data = self.data.dropna(thresh=value, axis=1)
        return self.data

    def fill_empty_values_with_0(self):
        self.data = self.data.fillna(0)
        return self.data


class DataSplit:
    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str,
                 random_state: int,
                 test_size: float) -> None:
        self.data = data
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size

    def main(self) -> tuple:
        x = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state)
        return x_train, x_test, y_train, y_test


class DataManipulation:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def main(self, stock_index: str) -> pd.DataFrame:
        self.data = self.datetime_to_date_index()
        self.data = self.filter_rows_by_index(stock_index)
        self.data = self.feature_columns()
        return self.data

    def datetime_to_date_index(self) -> pd.DataFrame:
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.set_index('Date')
        return self.data

    def filter_rows_by_index(self, stock_index) -> pd.DataFrame:
        self.data = self.data[self.data['Index'] == stock_index]
        return self.data    
    
    def feature_columns(self) -> list:
        self.data = self.data.drop(["Index", ], axis=1)
        return self.data.columns
    