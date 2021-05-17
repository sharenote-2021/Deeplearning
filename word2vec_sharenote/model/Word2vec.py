from gensim.models import Word2Vec


def word2vec(cluster_data):
    model = Word2Vec(cluster_data, size=100, window = 10, min_count=5, workers=5, iter=500, sg=1)
    # check embedding result
    word_vector = model.wv


    print(word_vector.vocab)
    # vocab 인덱스 초기화
    vocabulary = dict((t, i) for i, t in enumerate(word_vector.vocab))


    # word2vec 단어사전
    # 12434문장 -> 2568개 단어 사전
    print(len(vocabulary)) 
    print(word_vector.vocab['코로나'])

    #  코로나 : 72
    print(vocabulary["코로나"])

    return model, word_vector, vocabulary
