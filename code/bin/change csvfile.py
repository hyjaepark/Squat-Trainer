import pandas as pd
import numpy as np
import os

for i in ["test","train"]:
    df = pd.read_csv("C:\\Users\\user\\Desktop\\workspace\\workspace_new_image\\data\\{}_labels.csv".format(i))

    copy = df.copy()

    copy["filename"] = "flipped_" + copy["filename"].astype(str)
    copy["xmin"] = copy["width"]-copy["xmax"]
    copy["xmax"] = copy["width"]-df["xmin"]
    df_combined = df.append(copy)

    df_combined.to_csv("{}_labels.csv".format(i),index=False)