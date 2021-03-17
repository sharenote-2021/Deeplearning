import os
import pandas as pd
import numpy as np

def main(csv_list):

    tmp = []
    for j in csv_list:
        
        csv_file = pd.read_csv(j, names = ["date", "subject", "media", "title", "article", "link"], header = None)
        df = csv_file[["subject", "title"]]
        
        if df["subject"][0] == "정치":
            df["subject"] = 0
        elif df["subject"][0] == "경제":
            df["subject"] = 1
        elif df["subject"][0] == "사회":
            df["subject"] = 2
        
        # 각 카테고리 별로 3000개만 추출 
        tmp.append(df[:3000])

    # 카테고리 별 뉴스 병합
    df_1 = pd.concat([tmp[0], tmp[1]])
    df_1 = pd.concat([df_1, tmp[2]])

    df_1.reset_index(drop=True, inplace=True)
    df_1.astype({"subject" : int}).dtypes

    print(df_1['subject'])
    print(df_1.dtypes)

    df_1.to_csv("./category.csv")


if __name__ == "__main__":
    data_path = "./output"
    data_list = os.listdir(data_path)

    print(data_list)

    path_list = []
    for i in data_list:
        if i == ".DS_Store":
            continue
        else: 
            path_list.append(os.path.join(data_path, i))

    main(path_list)