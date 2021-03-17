import pandas as pd


import re

def test(sentence):
    # s='韓子는 싫고, 한글은 nice하다. English 쵝오 -_-ㅋㅑㅋㅑ ./?!'
    s = sentence
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    # hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
    result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    # print (result)
    return result

# # --------------------------데이터 로드--------------------


df = pd.read_csv("./category.csv")
news_title = df["title"]
subject = df['subject']

# # --------------------------데이터 전처리--------------------


data = []
raw_data = []
for sent in news_title:
    preprocess = test(sent)
    data.append(preprocess)
    raw_data.append(sent)


# --------------------------토크나이저 로드--------------------

import numpy as np 

from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.tokenizer import LTokenizer


word_extractor = WordExtractor(
    min_frequency=100, # example
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(news_title)
words = word_extractor.extract()

cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
tokenizer = LTokenizer(scores=cohesion_score)


# # --------------------------word2vec 데이터 전처리--------------------


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

from gensim.models import Word2Vec


model = Word2Vec(cluster_data, size=100, window = 3, min_count=5, workers=5, iter=500, sg=1)
# check embedding result
word_vector = model.wv


sentence_vectors = []
for sent in cluster_data:
    sent_vector = [0]*100
    for p in sent:
        # word2vec의 min_count 조건으로 인해 없는 단어가 있을 수 있음
        try: 
            x = word_vector.get_vector(p)
        except:
            continue
        sent_vector += x
    
    sentence_vectors.append(sent_vector)


print("comlpete vetorization")

# # --------------------------코사인 유사도 기반 가중치 행렬 구함--------------------


from sklearn.metrics.pairwise import cosine_similarity
distance_matrix = cosine_similarity(sentence_vectors, sentence_vectors)
print("complete weight array")

print(distance_matrix.shape)

# title = test("성동구서 확진자 9명 추가…4명은 증상발현 후 확진")
# print(title)

# # --------------------------다음 title에 해당하는 인덱스 추출--------------------


# indices = [i for i, s in enumerate(data) if title in s]
# print(indices)

# # 원하는 뉴스와 가지고 있는 문장들에 대해 코사인 유사도 곱을 하여 리스트 형태 값 -> [(0, 0.92), (1, 0.91) ........] 추출
# sim_scores = list(enumerate(distance_matrix[indices][0]))
# sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


# # 내림차순으로 되어있는 리스트에 대해 비슷한 뉴스 10개 추출
# sim_scores = sim_scores[1:11]

# news_indices = [i[0] for i in sim_scores]

# print(news_indices)
# for index, p in enumerate(news_indices):
#     print(data[p], sim_scores[index][1])


# 코사인 유사도에 따른 threshold가 몇 인지 체크할 것
# 예를 들면 코사인 유사도가 0.85이상인 문장들의 경우 유사한 경우가 많다던지...
# 클러스터 안의 문장 개수가 적어도 서비스를 하는 관점에서는 일정 임계치를 넘지 않는 값은 비슷한 문장이라 분류 하지 않게 되는 것

from pandas import Series, DataFrame

indices = [i for i in range(len(news_title))]

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
df.to_csv('./%s_result_%d_except.csv'%(threshold, except_count), sep=',', na_rep='NaN')

print("전체 데이터 갯수 : %d, 임계치 이하 데이터 갯수 : %d"%(len(news_title), except_count))