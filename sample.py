import os ,pandas as pd
path = os.path.join(os.getcwd(),"GA-FJSP")
print(path)
Dataset = pd.read_excel(os.path.join(path,"Dataset.xlsx"),)
#"C:\Users\WENG Lab\pyworks\GA-FJSP\Dataset.xlsx"
#raw_df = pd.read_excel(os.path.join(DATASET_DIR,'Dataset.xlsx'), sheet_name=SHEET_NAME)
print(Dataset.head())