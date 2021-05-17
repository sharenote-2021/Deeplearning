import pandas as pd

from tools.preprocess import *
from model.Word2vec import *


# 플라스크 서버의 결과 값 내뱉는 함수가 될 것

def main(data, news_title):

    tdm_vocab, _ = tdm_array(data)
    print("tdm array setting complete\n")

    cluster_data = data_tokenize(news_title, tdm_vocab)
    print("data tokenize complete\n")

    model, word_vector, vocabulary = word2vec(cluster_data)
    print("word2vec learning complete")

    vocab, tdm = tdm_array(data, vocabulary)



    ########################### 키워드 입력 부분 ######################
    keyword_index = vocabulary["코로나"]
    ###############################################################




    # (word2vec 단어, word2vec 임베딩 차원)
    # (411, 100)
    word_vector_list = [word_vector[v] for v in vocabulary]

    print(len(vocabulary), len(word_vector_list))
    print(len(tdm_vocab))

    print(model.most_similar(positive=["고객"], topn=10))

    calc_weight_matrix(word_vector_list, news_title, tdm, keyword_index)





if __name__ == "__main__":

    # data_path = "csv file path"
    # df = pd.read_csv(data_path)

    df = pd.read_csv("../category.csv")
    news_title = df["title"]
    print("total data length is %d\n"%len(news_title))
    subject = df['subject']



    # # --------------------------데이터 전처리--------------------


    data = []
    for sent in news_title:
        preprocess = test(sent)
        data.append(preprocess)



    main(data, news_title)