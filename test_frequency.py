import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import OrderedDict

def get_features(file1,file2):
    # combining the files in first repo
    output_file1 = open("java-output-1.0.0.txt","w")
    for file in glob.glob(file1):
        f = open(file,"r")
        output_file1.write(f.read())
    output_file1.close()

    # combining the files in second repo
    output_file2 = open("java-output-1.0.2.txt","w")
    for file in glob.glob(file2):
        f = open(file,"r")
        output_file2.write(f.read())
    output_file2.close()

    file1 = open("java-output-1.0.0.txt","r")
    file2 = open("java-output-1.0.2.txt","r")

    # removing non-alphabets
    file1 = file1.read().replace('/',' ').replace('#',' ').replace('<',' ').replace('""',' ').replace('_',' ').replace(':',' ').replace('+',' ').translate(str.maketrans('','','0123456789'))
    file2 = file2.read().replace('/',' ').replace('#',' ').replace('<',' ').replace('""',' ').replace('_',' ').replace(':',' ').replace('+',' ').translate(str.maketrans('','','0123456789'))

    vectorizer = CountVectorizer()
    corpus = [file1,file2]
    X = vectorizer.fit_transform(corpus)
    output_features = vectorizer.get_feature_names() # total list of features

    vector1 = np.array(X.toarray()[0])  # vector array for file1
    vector2 = np.array(X.toarray()[1])  # vector array for file2

    difference_vector = np.subtract(vector1,vector2)
    difference_vector = np.absolute(difference_vector)

    # dictionary of features and difference vector
    dictt = dict(zip(output_features,difference_vector))

    # sort in ascending order
    dictt = OrderedDict(sorted(dictt.items(),key=lambda kv:kv[1]))
    bottom25 = {k: dictt[k] for k in list(dictt)[:25]}
    bottom35 = {k: dictt[k] for k in list(dictt)[:35]}

    # sort in descending order
    dictt = OrderedDict(sorted(dictt.items(),key=lambda kv:kv[1],reverse=True))
    top25 = {k: dictt[k] for k in list(dictt)[:25]}
    top35 = {k: dictt[k] for k in list(dictt)[:35]}

    return (top25,bottom25,top35,bottom35)

def main():
    # test with py files
    #file1="repositories/python/nltk-3.2.4/nltk/*.py"
    #file2="repositories/python/nltk-3.4.4/nltk/*.py"

    # test with java files
    file1 = "repositories/java/Fast-Android-Networking-1.0.0/app/src/main/java/com/networking/*.java"
    file2 = "repositories/java/Fast-Android-Networking-1.0.2/app/src/main/java/com/networking/*.java"

    (top25,bottom25,top35,bottom35) = get_features(file1,file2)

    print("Top 25 features",top25)
    print("Bottom 25 features",bottom25)
    print("Top 35 features",top35)
    print("Bottom 35 features",bottom35)

main()


####### sample output for python file
#Top 25 features {'self': 44, 'rule': 33, 'in': 27, 'defaultdict': 24, 'the': 24, 'concordance': 23, 'for': 22, 'grammar': 22, 'if': 22, 'ax': 19, 'return': 19, 'line': 17, 'pylab': 17, 'new': 16, 'print': 16, 'word': 16, 'list': 15, 'seed': 14, 'append': 13, 'def': 13, 'result': 13, 'text': 13, 'len': 12, 'dict': 11, 'param': 11}
#Bottom 25 features {'aaa': 0, 'abbb': 0, 'abbbc': 0, 'abbr': 0, 'abbreviate': 0, 'abc': 0, 'abcmeta': 0, 'able': 0, 'abort': 0, 'aborted': 0, 'aborting': 0, 'about': 0, 'above': 0, 'abs': 0, 'abspath': 0, 'abstract': 0, 'abstractcollocationfinder': 0, 'abstractlazysequence': 0, 'abstractmethod': 0, 'abstractparentedtree': 0, 'ac': 0, 'academic': 0, 'accelerator': 0, 'accents': 0, 'acceptable': 0}
#Top 35 features {'self': 44, 'rule': 33, 'in': 27, 'defaultdict': 24, 'the': 24, 'concordance': 23, 'for': 22, 'grammar': 22, 'if': 22, 'ax': 19, 'return': 19, 'line': 17, 'pylab': 17, 'new': 16, 'print': 16, 'word': 16, 'list': 15, 'seed': 14, 'append': 13, 'def': 13, 'result': 13, 'text': 13, 'len': 12, 'dict': 11, 'param': 11, 'start': 11, 'freqs': 10, 'lines': 10, 'of': 10, 'random': 10, 'resource': 10, 'rhs': 10, 'type': 10, 'context': 9, 'data': 9}
#Bottom 35 features {'aaa': 0, 'abbb': 0, 'abbbc': 0, 'abbr': 0, 'abbreviate': 0, 'abc': 0, 'abcmeta': 0, 'able': 0, 'abort': 0, 'aborted': 0, 'aborting': 0, 'about': 0, 'above': 0, 'abs': 0, 'abspath': 0, 'abstract': 0, 'abstractcollocationfinder': 0, 'abstractlazysequence': 0, 'abstractmethod': 0, 'abstractparentedtree': 0, 'ac': 0, 'academic': 0, 'accelerator': 0, 'accents': 0, 'acceptable': 0, 'accepted': 0, 'access': 0, 'accessed': 0, 'accessor': 0, 'accidentally': 0, 'according': 0, 'account': 0, 'accuracy': 0, 'accurate': 0, 'acl': 0}

# I can see that the distribution first varies a lot between the more different features then starts to decrease (from top 25 features to 35 features,
# there is more distribution between the top 25 rather than from 26th to 35th

####### sample output for java file
#Top 25 features {'response': 8, 'anerror': 6, 'public': 6, 'tag': 6, 'void': 6, 'androidnetworking': 4, 'isrequestrunning': 4, 'log': 4, 'override': 4, 'view': 4, 'file': 3, 'api': 2, 'build': 2, 'cancel': 2, 'checkoptionsrequest': 2, 'com': 2, 'final': 2, 'getasokhttpresponse': 2, 'github': 2, 'headers': 2, 'https': 2, 'issues': 2, 'key': 2, 'logerror': 2, 'new': 2}
#Bottom 25 features {'activity': 0, 'add': 0, 'addbodyparameter': 0, 'addheaders': 0, 'addjsonobjectbody': 0, 'addmultipartfile': 0, 'addpathparameter': 0, 'addqueryparameter': 0, 'agreed': 0, 'amit': 0, 'amitshekhar': 0, 'an': 0, 'analyticslistener': 0, 'and': 0, 'android': 0, 'animageloader': 0, 'animageview': 0, 'anrequest': 0, 'anresponse': 0, 'any': 0, 'apache': 0, 'apiendpoint': 0, 'apitestactivity': 0, 'app': 0, 'appcompatactivity': 0}
#Top 35 features {'response': 8, 'anerror': 6, 'public': 6, 'tag': 6, 'void': 6, 'androidnetworking': 4, 'isrequestrunning': 4, 'log': 4, 'override': 4, 'view': 4, 'file': 3, 'api': 2, 'build': 2, 'cancel': 2, 'checkoptionsrequest': 2, 'com': 2, 'final': 2, 'getasokhttpresponse': 2, 'github': 2, 'headers': 2, 'https': 2, 'issues': 2, 'key': 2, 'logerror': 2, 'new': 2, 'okhttp': 2, 'okhttpresponselistener': 2, 'onerror': 2, 'onresponse': 2, 'options': 2, 'square': 2, 'this': 2, 'tostring': 2, 'utils': 2, 'after': 1}
#Bottom 35 features {'activity': 0, 'add': 0, 'addbodyparameter': 0, 'addheaders': 0, 'addjsonobjectbody': 0, 'addmultipartfile': 0, 'addpathparameter': 0, 'addqueryparameter': 0, 'agreed': 0, 'amit': 0, 'amitshekhar': 0, 'an': 0, 'analyticslistener': 0, 'and': 0, 'android': 0, 'animageloader': 0, 'animageview': 0, 'anrequest': 0, 'anresponse': 0, 'any': 0, 'apache': 0, 'apiendpoint': 0, 'apitestactivity': 0, 'app': 0, 'appcompatactivity': 0, 'append': 0, 'appinstance': 0, 'applicable': 0, 'application': 0, 'argb': 0, 'array': 0, 'as': 0, 'at': 0, 'atbxix': 0, 'base': 0}


# I notice that in Python there is more distributional similarity between code versions whereas in java is way less.
# The reason could be that Python is higher level programming than Java or C++