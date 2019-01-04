# _*_ coding: UTF-8 _*_
import pandas as pd

CUSTOM_FAQ = "t_custom_tag.xlsx"

df = pd.read_excel(CUSTOM_FAQ)
print(df)
