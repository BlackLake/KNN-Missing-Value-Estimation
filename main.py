import pandas as pd

xl_file = pd.ExcelFile("energy-efficiency.xlsx")

dfs = pd.read_excel(xl_file, sheet_name="Sheet1", header=None)


print(dfs)
