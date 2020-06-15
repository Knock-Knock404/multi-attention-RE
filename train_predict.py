from load import loaddata
from model import myModel

import os
from collections import Counter

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences


def get_Result(y,g=False):
    result = np.zeros(shape=y.shape)
    if g is False:
        for i in range(y.shape[0]):
            if y[i,0] >= y[i,1]:
                result[i,0] = 1
            else:
                result[i,1] = 1
    else:
        for i in range(y.shape[0]):
            if y[i,0] >= g:
                result[i,0] = 1
            else:
                result[i,1] = 1

    return result


def get_F1Value(y_,y,path,return_matrix=False,show_matrix=True):
    matrix = np.zeros(shape=(2,2),dtype="int32")
    for i in range(y.shape[0]):
        if y[i,0] == y_[i,0] and y[i,0] == 1:  # TP
            matrix[0,0] += 1
        if y[i,0] == y_[i,0] and y[i,0] == 0:  # TN
            matrix[1,1] += 1
        if y[i,0] == 0 and y_[i,0] == 1:  # FN
            matrix[1,0] += 1
        if y[i,0] == 1 and y_[i,0] == 0:  # FP
            matrix[0,1] += 1
    P = matrix[0,0]/(matrix[0,0]+matrix[1,0])
    R = matrix[0,0]/(matrix[0,0]+matrix[0,1])
    F1 = 2*P*R/(P+R)

    file = open(path+"/"+"result.txt","w",encoding="utf-8")
    file.write(str(matrix)+"\n")
    file.write("F: "+str(F1)+" P: "+str(P)+" R: "+str(R)+"\n")
    file.close()

    if show_matrix is True:
        print(matrix)
    print("F: ",F1," P: ",P," R: ",R)

    if return_matrix is True:
        return matrix


def get_ROC(y_,y,path,gate=10,):
    import matplotlib.pyplot as plt
    file = open(path+"/"+"result.txt","w",encoding="utf-8")
    roc_x,roc_y = [],[]
    result = [[y_[i,0],y[i,0]]for i in range(y.shape[0])]
    result = sorted(result,key=lambda x:x[0],reverse=True)

    gate = [i/gate for i in range(1,gate)]
    for g in gate:
        print("its gate :  ",g)
        matrix = np.zeros(shape=(2,2),dtype="int32")
        for r in result:
            if r[0] >= g and r[1] == 1:  # TP
                matrix[0,0] += 1
            if r[0] < g and r[1] == 0:  # TN
                matrix[1,1] += 1
            if r[0] < g and r[1] == 1:  # FN
                matrix[1,0] += 1
            if r[0] >= g and r[1] == 0:  # FP
                matrix[0,1] += 1

        tpr = matrix[0,0]/(matrix[0,0]+matrix[1,0])
        fpr = matrix[0,1]/(matrix[0,1]+matrix[1,1])
        roc_x.append(fpr)
        roc_y.append(tpr)

        P = matrix[0,0]/(matrix[0,0]+matrix[1,0])
        R = matrix[0,0]/(matrix[0,0]+matrix[0,1])
        F1 = 2*P*R/(P+R)
        file.write("F: "+str(F1)+" P: "+str(P)+" R: "+str(R)+" gate: "+str(g)+"\n")
        print(matrix)
        print("tpr: ",tpr," fpr: ",fpr)
        print("F: ",F1," P: ",P," R: ",R)

    file.close()

    plt.plot(roc_x,roc_y)
    plt.savefig(path+"/ROCfig.png")
    plt.show()


def main(path):
    train,test,wordVocab = loaddata(path)

    maxSentLen = 140
    train_sent = pad_sequences(train["sent"],maxlen=maxSentLen,padding="post")
    train_gen = pad_sequences(train["gen"],maxlen=5,padding="post")
    train_disease = pad_sequences(train["disease"],maxlen=5,padding="post")
    test_sent = pad_sequences(test["sent"],maxlen=maxSentLen,padding="post")
    test_gen = pad_sequences(test["gen"],maxlen=5,padding="post")
    test_disease = pad_sequences(test["disease"],maxlen=5,padding="post")
    train_rel = np.array(train["rel"])
    test_rel = np.array(test["rel"])

    print(train_rel.shape)
    print(test_rel.shape)

    # load pre-embedding
    print("loading embedding...")
    embedding_size = 200
    embeddingVocab_size = len(wordVocab)

    # w2v_dir_path = "/media/network/watching_dog/embedding/bio_nlp_vec/PubMed-shuffle-win-30.bin"
    w2v_dir_path = "/media/kazane/watching_dog/embedding/bio_nlp_vec/PubMed-shuffle-win-30.bin"

    word2vec = KeyedVectors.load_word2vec_format(w2v_dir_path, binary=True, unicode_errors='ignore')

    print("build embedding weights...")
    embedding_weights = np.zeros((embeddingVocab_size + 1, embedding_size))
    unknow_words = []
    know_words = []
    for word, index in wordVocab.items():
        try:
            embedding_weights[index, :] = word2vec[word.lower()]
            know_words.append(word)
        except KeyError as E:
            # print(E)
            unknow_words.append(word)
            embedding_weights[index, :] = np.random.uniform(-0.025, 0.025, embedding_size)
    print("unknow_per: ", len(unknow_words) / embeddingVocab_size, " unkownwords: ", len(unknow_words), " vocab_size: ",
          embeddingVocab_size)

    model = myModel(sent_lenth=maxSentLen,word_embedding=embedding_weights)
    model.attn(embedding=embedding_weights)

    model.train(inputs=[train_sent,train_gen,train_disease,],
                label=[train_rel,],
                save_path="./outputs/checkpoint",
                validation_split=0.1,
                batch_size=128,
                epochs=5,
                )
    y_ = model.predict([test_sent,test_gen,test_disease])
    # np.save(file=path+"/"+"outputs.npy",arr=y_)

    is_gate = False
    if is_gate is False:
        y_ = get_Result(y_,)
        get_F1Value(y_=y_,y=test_rel,path=path)
    else:
        get_ROC(y_,test_rel,path=path,gate=100)


if __name__ == "__main__":
    main("./data/euadr")
    main("./data/GAD")
