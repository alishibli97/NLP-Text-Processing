# Author: Ali Shibli
# EECE 634: Natural Language Processing
# This code generates an nxn matrix where n is the number of features of the input files \
# and computes the euclidean distance between those features
# Each row represents a vector of co-occurring words in each line in the input file
# NOTE
#       this code computes 17 lines per second for python code in the txt file
#       this code computes 16 lines per second for java code in the txt file
#       runs where made on intel core i7 9th gen

from numba import jit,cuda,vectorize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import time,datetime

#@vectorize(['string(string, string)'], target='cuda')
#@jit(target="cuda")
#@vectorize(['float32(float32, float32)'], target='cuda')

def main():
    file1 = open("python-subtext-3.2.4.txt", "r")
    file2 = open("python-subtext-3.4.4.txt", "r")

    file1 = file1.read().replace('/', ' ').replace('#', ' ').replace('<', ' ').replace('""', ' ').replace('_',' ').replace(':', ' ').replace('+', ' ').translate(str.maketrans('', '', '0123456789')).replace('(', ' ').replace(')',' ').replace('*', ' ').replace('{', ' ').replace('}', ' ').replace('"', ' ').replace(';', ' ').replace(',', ' ').replace('@',' ').replace('.', ' ')
    file2 = file2.read().replace('/', ' ').replace('#', ' ').replace('<', ' ').replace('""', ' ').replace('_',' ').replace(':', ' ').replace('+', ' ').translate(str.maketrans('', '', '0123456789')).replace('(', ' ').replace(')',' ').replace('*', ' ').replace('{', ' ').replace('}', ' ').replace('"', ' ').replace(';', ' ').replace(',', ' ').replace('@',' ').replace('.', ' ')

    vectorizer = CountVectorizer()
    corpus = [file1,file2]
    X = vectorizer.fit_transform(corpus)
    output_features = vectorizer.get_feature_names() # total list of features/tokens

    lines1 = file1.splitlines()
    lines2 = file2.splitlines()

    dict1 = {}
    for i in range(len(lines1)):
        lines1[i]=lines1[i].strip()
        words = lines1[i].split()
        for j in range(len(words)):
            dict1[(i,j)] = words[j]

    dict2 = {}
    for i in range(len(lines2)):
        lines2[i]=lines2[i].strip()
        words = lines2[i].split()
        for j in range(len(words)):
            dict2[(i,j)] = words[j]

    # output dictionary of vectors for each word
    dict_sim1 = dict(zip(output_features,[ [0 for i in range(len(output_features))] for i in range(len(output_features))]))
    dict_sim2 = dict(zip(output_features,[ [0 for i in range(len(output_features))] for i in range(len(output_features))]))

    print("Started first computation at",datetime.datetime.now())
    start = time.time()
    for token in dict_sim1.keys():
        for tokenP in dict_sim1.keys():
            for line in lines1:
                words = line.strip().split()
                if token in words and tokenP in words:
                    difference = words.index(tokenP)-words.index(token)
                    if (abs(difference)<=3):
                        #print("The difference is",difference, "I am in line number number",lines1.index(line))
                        dict_sim1[token][output_features.index(tokenP)] +=1
    end = time.time()
    print("done computation1 with elapsed time",end-start)

    print("started second computation at",datetime.datetime.now())
    start = time.time()
    for token in dict_sim2.keys():
        for tokenP in dict_sim2.keys():
            for line in lines2:
                words = line.strip().split()
                if token in words and tokenP in words:
                    difference = words.index(tokenP)-words.index(token)
                    if (abs(difference)<=3):
                        #print("The difference is",difference, "I am in line number number",lines1.index(line))
                        dict_sim2[token][output_features.index(tokenP)] +=1
    end = time.time()
    print("done computation2 with elapsed time",end-start)

    arrays1 = np.array(list(dict_sim1.values()))
    arrays2 = np.array(list(dict_sim2.values()))
    euclidean_distance = np.linalg.norm(arrays1-arrays2)
    print("The euclidean distance vector between the two files is",euclidean_distance)

    df1 = pd.DataFrame([dict_sim1], index=[0])
    df2 = pd.DataFrame([dict_sim2], index=[0])
    df1 = (df1.T)
    df2 = (df2.T)

    # writing to output files
    df1.to_excel('repositories/python/dict_sim1.xlsx')
    df2.to_excel('repositories/python/dict_sim2.xlsx')

    #df1.to_excel('repositories/java/dict_sim1.xlsx')
    #df2.to_excel('repositories/java/dict_sim2.xlsx')


main()


#### RESULTS

## Started first computation at 2019-10-09 23:56:07.184556
# done computation1 with elapsed time 221.769305229187
# started second computation at 2019-10-09 23:59:48.953861
# done computation2 with elapsed time 198.13432240486145
# The euclidean distance vector between the two files is 85.67379996241559

## sample run for java code
# Started first computation at  2019-10-09 22:14:08.231255
# done computation1 with elapsed time 162.6667275428772
# started second computation at 2019-10-09 22:16:50.897982
# done computation2 with elapsed time 163.8318178653717
# The euclidean distance vector between the two files is 20.54263858417414


## Conclusion
# Even though python scipt was shorter, there was so much more difference
# Higher level programming has more distribution