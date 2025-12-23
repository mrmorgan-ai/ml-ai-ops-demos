import pandas as pd #type: ignore
import time

data_path = r"C:\Users\jhoni\Documents\LooperAI\repositorios\ml-ai-ops-demos\customer_churn_prediction\data\01-raw\creditcard.csv"

df_raw_data = pd.read_csv(data_path,header=0)
print(df_raw_data.head(5))
