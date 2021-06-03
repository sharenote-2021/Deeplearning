import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Series, DataFrame

from tools.preprocess import *
from model.Word2vec import *
from db_connect import connect_db

# 플라스크 서버의 결과 값 내뱉는 함수가 될 것

def main(data, news_title):


    # --------------------------토크나이저 로드--------------------

    tokenizer = data_tokenize(news_title)
    print("data tokenize complete\n")


    # --------------------------word2vec 데이터 전처리--------------------


    cluster_data = []

    for k, title in enumerate(news_title):
        title = test(title)
        sent = tokenizer.tokenize(title, flatten=False)
        sentence = []
        # sent -> ['단어', ''] 
        for i in sent:
            sentence.append(i[0])

        cluster_data.append(sentence)

    # --------------------------Word2Vec embedding--------------------


    sentence_vectors = word2vec(cluster_data)
    print("word2vec learning complete")

    # --------------------------코사인 유사도 기반 가중치 행렬 구함--------------------

    distance_matrix = cosine_similarity(sentence_vectors, sentence_vectors)
    print("complete weight array")

    print(distance_matrix.shape)


    indices = [i for i in range(len(news_title))]
    print(len(indices))
    tmp = []

    save_data = {"기준뉴스문장" : [], "임계치문장" : [], "코사인유사도":[], "클러스터갯수": []}

    except_count = 0
    threshold = 0.7

    for i in indices:

        sim_scores = list(enumerate(distance_matrix[[i]][0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:]
        k = [o for o in sim_scores if o[1] > threshold] 

        if len(k) == 0 :
            except_count += 1
            continue
        else :
            save_data["기준뉴스문장"].append(raw_data[i])
            save_data["임계치문장"].append(raw_data[k[-1][0]])
            save_data["코사인유사도"].append(k[-1][1])
            save_data["클러스터갯수"].append(len(k))
            

    df = DataFrame(save_data)
    df = df.sort_values(by=['클러스터갯수'],ascending=False)
    df = df.drop_duplicates(['기준뉴스문장'])
    df = df.drop_duplicates(['코사인유사도'])

    df.to_csv('./output/%s_result_%d_except.csv'%(threshold, except_count), sep=',', na_rep='NaN')

    print("전체 데이터 갯수 : %d, 임계치 이하 데이터 갯수 : %d"%(len(news_title), except_count))


if __name__ == "__main__":

    # data_path = "csv file path"
    # df = pd.read_csv(data_path)

    if os.path.exists('./db_connect.p'):
        df = connect_db()
        print(df)
        news_title = df['topic_news_title']
    else:
        df = pd.read_csv("./category.csv")
        news_title = df["title"]
    print("total data length is %d\n"%len(news_title))


    # # --------------------------데이터 전처리--------------------


    data = []
    raw_data = []
    for sent in news_title:
        preprocess = test(sent)
        data.append(preprocess)
        raw_data.append(sent)



    main(data, news_title)