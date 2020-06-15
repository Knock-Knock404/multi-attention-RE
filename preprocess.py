import os
import re
import pandas as pd
import numpy as np

def sentCleaner(sent):
    sent = re.sub(r"\.$","",sent)
    sent = re.sub(r"-"," - ",sent)
    sent = re.sub(r"\.\s"," . ",sent)
    sent = re.sub(r","," , ",sent)
    sent = re.sub(r"<"," ( ",sent)
    sent = re.sub(r">"," ( ",sent)
    sent = re.sub(r"\("," ( ",sent)
    sent = re.sub(r"\)"," ) ",sent)
    sent = re.sub(r"/"," : ",sent)
    sent = re.sub(r":"," : ",sent)
    sent = re.sub(r";"," : ",sent)
    sent = re.sub(r"%"," % ",sent)
    sent = re.sub(r"\["," [ ",sent)
    sent = re.sub(r"\]"," ] ",sent)
    sent = re.sub(r"="," = ",sent)
    return sent


def getSentWithRel(data):
    # process sent
    sents = []
    for sent in data["sent"]:
        sent = sentCleaner(sent)
        sents.append(sent.split())

    # process rel
    rels = []
    for rel in data["rel"]:
        if str(rel) == "1": rels.append([1,0])
        if str(rel) == "0": rels.append([0,1])

    return {"sent":sents,"rel":rels}


def loadtxt(path):
    sents,rels = [],[]
    f = open(path,"r",encoding="utf-8")
    if "test" in path: f.readline()
    while True:
        try:
            l = f.readline()
            l = l.split("\t")
            sents.append(l[-2])
            rels.append(l[-1].strip())
        except Exception as E:
            break
    # for l in f.readlines():
    #     l = l.split("\t")
    #     sents.append(l[-2])
    #     rels.append(l[-1].strip())
    f.close()
    return sents,rels


def readtxt(path):
    train = {"sent":[],"rel":[]}
    dev = {"sent":[],"rel":[]}
    test = {"sent":[],"rel":[]}
    for d in os.listdir(path):
        for f_list in [os.listdir(path+"/"+d)]:
            for f in f_list:
                try:
                    if "train" in f:
                        sent,rel = loadtxt(path+"/"+d+"/"+f)
                        train["sent"] += sent
                        train["rel"] += rel
                    elif "dev" in f:
                        sent,rel = loadtxt(path+"/"+d+"/"+f)
                        dev["sent"] += sent
                        dev["rel"] += rel
                    elif "test" in f:
                        sent,rel = loadtxt(path+"/"+d+"/"+f)
                        test["sent"] += sent
                        test["rel"] += rel
                    else:
                        print("file error")
                        exit()
                except Exception as E:
                    print(E)
                    pass
    return train,test


if __name__ == "__main__":
    train,test = readtxt("./data/euadr")
    # print(test["rel"])
    train = getSentWithRel(train)
    test = getSentWithRel(test)
    # print(test["rel"])