from preprocess import readtxt,getSentWithRel
from collections import Counter
import re
import numpy as np


def wordCleaner(word):
    if re.search(r"^\d+\.\d+$",word) or str(word).isdigit():
        return "<num>"
    else:
        return str(word).lower().strip()

def oneHot3D(matrix,vocab=False,returnvocab=False):
    '''
    one-hot encoding for char
    '''

    # creat vocab
    if vocab is False:
        counter = Counter()
        for line in matrix:
            for token in line:
                for char in token:
                    try:
                        counter[str(char)]+=1
                    except ValueError as VE:
                        pass
        counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
        vocab = {v[0]:i+1 for i,v in enumerate(counter)}

    # replace token to token-id
    matrixTags = []
    for l,line in enumerate(matrix):
        lineTag = []
        for t,token in enumerate(line):

            # sentences output for 3D
            chars = []
            for char in token:
                try:
                    chars.append(vocab[char])
                except KeyError as KE:
                    print("3d: ",KE)
                    exit()
            while len(chars) < 33:  # maxCharLenth is 33
                chars.append(0)  # 0 as padding
            lineTag += chars
        matrixTags.append(lineTag)

    # return vocab
    if returnvocab:
        return matrixTags,vocab
    return matrixTags


def oneHot2D(matrix,vocab=None,returnvocab=False):
    '''
    one-hot encoding for word / pos
    '''

    # creat vocab
    if vocab is None:
        counter = Counter()
        for line in matrix:
            for token in line:
                try:
                    counter[token]+=1
                except KeyError as KE:
                    pass
        counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
        vocab = {v[0]:i+1 for i,v in enumerate(counter)}

    # replace token to token-id
    matrixTags = []
    for l,line in enumerate(matrix):
        lineTag = []
        for t,token in enumerate(line):
            try:
                lineTag.append(vocab[token])
            except KeyError as KE:
                lineTag.append(vocab["<unknow>"])
                pass
        matrixTags.append(lineTag)

    # return vocab
    if returnvocab:
        return matrixTags,vocab
    return matrixTags


def oneHot1D(matrix,vocab=False,returnvocab=False):
    '''
    one-hot encoding for word / pos
    '''

    # creat vocab
    if vocab is False:
        counter = Counter()
        for token in matrix:
            try:
                counter[token]+=1
            except KeyError as KE:
                pass
        counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
        vocab = {v[0]:i+1 for i,v in enumerate(counter)}

    # replace token to token-id
    matrixTag = []
    for w,word in enumerate(matrix):
        try:
            matrixTag.append(vocab[word])
        except KeyError as KE:
            exit()
            pass

    # return vocab
    if returnvocab:
        return matrixTag,vocab
    return matrixTag


def sentClean(sents):
    sentTags = []
    for sent in sents:
        sentTag = []
        for word in sent:
            sentTag.append(wordCleaner(word))
        sentTags.append(sentTag)
    return sentTags


def getVocab(matrix,ndim=2,isClean=False,isUnknow=False):
    if isClean is True:
        matrix = sentClean(matrix)

    counter = Counter()
    if ndim == 1:
        for token in matrix:
            try:
                counter[token]+=1
            except KeyError as KE:
                print("vocab1Derror:",KE)
                exit()
    elif ndim == 2:
        for line in matrix:
            for token in line:
                try:
                    counter[token]+=1
                except KeyError as KE:
                    print("vocab2Derror:",KE)
                    exit()
    elif ndim == 3:
        for line in matrix:
            for token in line:
                for char in token:
                    try:
                        counter[str(char)]+=1
                    except KeyError as KE:
                        print("vocab2Derror:",KE)
                        exit()
    counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
    if isUnknow:
        vocab = {str(v[0]):i+2 for i,v in enumerate(counter)}
        vocab["<unknow>"] = 1
        return vocab  # 0 as padding,1 as unknow
    return {str(v[0]):i+1 for i,v in enumerate(counter)}  # 0 as padding,1 as unknow


def getGenEmbedding(sents,vocab):
    genEmbedding,disEmbedding = [],[]
    for sent in sents:
        gens,diseas = [],[]
        for w,word in enumerate(sent):
            if word == vocab["@gene$"]:
                for i in range(3):
                    try:
                        gens.append(sent[w-2+i])
                    except IndexError as IE:
                        gens.append(0)
                        continue
                for i in range(2):
                    try:
                        gens.append(sent[w+1+i])
                    except IndexError as IE:
                        gens.append(0)
                        continue
            elif word == vocab["@disease$"]:
                for i in range(3):
                    try:
                        diseas.append(sent[w-3+i])
                    except IndexError as IE:
                        diseas.append(0)
                        continue
                for i in range(2):
                    try:
                        diseas.append(sent[w+1+i])
                    except IndexError as IE:
                        diseas.append(0)
                        continue
            else:
                pass
        genEmbedding.append(gens)
        disEmbedding.append(diseas)

    return genEmbedding,disEmbedding


def getPosEmbedding():
    pass


def loaddata(path):
    train,test = readtxt(path)
    train = getSentWithRel(train)
    test = getSentWithRel(test)
    train["sent"] = sentClean(train["sent"])
    test["sent"] = sentClean(test["sent"])

    # prepar for word
    wordVocab = getVocab(train["sent"]+test["sent"])
    train["sent"] = oneHot2D(train["sent"],vocab=wordVocab)
    test["sent"] = oneHot2D(test["sent"],vocab=wordVocab)

    # prepar for window
    train["gen"],train["disease"] = getGenEmbedding(train["sent"],vocab=wordVocab)
    test["gen"],test["disease"] = getGenEmbedding(test["sent"],vocab=wordVocab)

    return train,test,wordVocab


if __name__ == "__main__":
    train,test,wordVocab = loaddata("./data/euadr")
    print(train["gen"][0])

    # for analysis
    maxlen = []
    for s in train["sent"]+test['sent']:
        maxlen.append(len(s))
    print(max(maxlen),min(maxlen),np.mean(maxlen),np.median(maxlen))  # 140,19,35,29
