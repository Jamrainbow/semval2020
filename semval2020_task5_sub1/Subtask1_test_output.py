import csv
import pandas as pd
from collections import Counter
from tkinter import _flatten
import numpy as np
import gensim
from nltk.corpus import stopwords
import nltk
import os
from numpy.linalg import inv
from math import floor
LabeledSentence = gensim.models.doc2vec.LabeledSentence
TaggedDocument = gensim.models.doc2vec.TaggedDocument


english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-',"''","'",'``']
stoplist = stopwords.words('english')
stoplist = stoplist[0:44]+stoplist[60:128]+stoplist[136:143]
stoplist = stoplist + english_punctuations
print(stoplist.index(','),'is ,')
#print(stoplist)

validate_list =[]
validsetid = []
validsetdata = []
right_set = []
wrong_set = []
shufflelist = []
parses_train = []
parses_valid = []
parses_train2 = []
parses_valid2 = []
#parses = []
def get_x_data(trainsen,train_label,testsen,test_label):
    trainlabelized = []
    testlabelized = []
    for i, v in zip(train_label,trainsen):
        label = '%s' % (i)
        trainlabelized.append(TaggedDocument(v, tags = [label]))

    for i, v in zip(test_label, testsen):
        label = '%s' % (i)
        testlabelized.append(TaggedDocument(v, tags = [label]))
    return trainlabelized,testlabelized

def get_vec(trainlabelized,testlabelized,set_size):

    model = gensim.models.Doc2Vec(trainlabelized+testlabelized,vector_size=set_size, dm=1, window=4)
    #model.build_vocab(np.concatenate((trainlabelized,testlabelized)))
    sentence_vec = np.concatenate([np.array(model.docvecs[sen.tags[0]].reshape(1, set_size)) for sen in trainlabelized])
    tsence_vec = np.concatenate([np.array(model.docvecs[sen.tags[0]].reshape(1, set_size)) for sen in testlabelized])
    model.save('doc2vec30000.model')

    return sentence_vec, tsence_vec

def del_nonuse(data):
    tokens = []
    row_num = 0
    for l in data:
        tokens.append([])
        for word in l:
            if word not in stoplist:
                tokens[row_num].append(word)
        row_num = row_num + 1
    return tokens

def to_txt2(data):
    output = open('output2.txt', 'w', encoding='utf-8')
    for l in data:
        output.write(l)
        output.write('\n')
    output.close()


def to_txt(data):
    output = open('output2.txt','w+',encoding='utf-8')
    for l in data:
        for word in l:
            output.write(word)
            output.write(' ')
        output.write('\n')
    output.close()

def nltk_parse(data):
    tokens = []
    for sen in data:
        tokens.append(nltk.word_tokenize(sen))
    return tokens

def dataprocess_y(rawData):
    df_y = rawData['gold_label']
    data_y = pd.DataFrame(df_y.apply(int), columns=['gold_label'])
    return df_y


def sigmoid(z):
    res = 1/(1.0+np.exp(-z))
    return res
    #return np.clip(res,1e-8,(1-(1e-8)))

def _shuffle(X, Y, parses):                                 #X and Y are np.array
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)#打乱行号
    #print(randomize)
    #shufflelist = randomize
    parses_train = parses[randomize]
    parses_valid = parses[randomize]

    return (X[randomize], Y[randomize], parses_train) #返回打乱行号后的数据


def split_valid_set(X, Y, percentage, parses):
    all_size = X.shape[0]  # 矩阵的行
    valid_size = int(floor(all_size * percentage))  # floor向下取整”，或者说“向下舍入”
    #print(parses)

    X, Y, parses_train = _shuffle(X, Y, parses)
    #print('parses_train is', parses_train)
    X_valid, Y_valid = X[:valid_size], Y[:valid_size]
    X_train, Y_train = X[valid_size:], Y[valid_size:]
    parses_train2,parses_valid2 = parses_train[valid_size:],parses_valid[:valid_size]
    print('valid_size is ',str(valid_size))
    print('parses size is',str(np.array(parses_train).shape[0]))
    #print('parses_train2 is',parses_train2)
    validsetid.append(valid_size)
    return X_train, Y_train, X_valid, Y_valid, parses_train2




def valid(X,Y,mu1,mu2,shared_sigma,N1,N2,feature_size):
    sigma_inv=inv(shared_sigma) #相同的协方差即可
    w=np.dot((mu1-mu2),sigma_inv)
    X_t=X.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(float(N1)/N2)
    a=np.dot(w,X_t)+b
    y=sigmoid(a)
    print('valid falses display')
    #dis_false_answer(y, Y, parses)
    y_=np.around(y)#返回四舍五入后的值
    #print('calculate y is ',y_)
    #print('true y is ',Y)
    result=(np.squeeze(Y)==y_)
    #spilt_right_wrong(y_, Y, result)
    #print('the result is ',result)
    print('when feature_size is ',str(feature_size),' Valid acc =%f '%(float(result.sum())/result.shape[0]))
    validate_list.append([feature_size,(float(result.sum())/result.shape[0])])
    get_presion_recall(y_, Y)

    n_row = 0
    needlist = []
    for pred,p in zip( y,y_):
        needlist.append([])
        needlist[n_row].append(pred)
        needlist[n_row].append(p)
        n_row = n_row + 1

    df2 = pd.DataFrame(needlist)
    df2.rename(columns={0: 'pred_label'}, inplace=True)
    print(df2)
    df2.to_csv(os.path.join('subtask1_valid.csv'), index=False)

    return

def valid_on_train(X,Y,mu1,mu2,shared_sigma,N1,N2,feature_size, parses_train2):
    sigma_inv = inv(shared_sigma)  # 相同的协方差即可
    w = np.dot((mu1 - mu2), sigma_inv)
    X_t = X.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    print('train falses display')
    #dis_false_answer(y,Y,parses_train2)
    y_ = np.around(y)  # 返回四舍五入后的值
    result = (np.squeeze(Y) == y_)
    #spilt_right_wrong(y_,Y,result)
    print('when feature_size is ', str(feature_size), ' Valid_on_train acc =%f ' % (float(result.sum()) / result.shape[0]))
    validate_list.append([feature_size, (float(result.sum()) / result.shape[0])])
    get_presion_recall(y_,Y)
    return


def get_presion_recall(Y, y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i,j in zip(Y,y):
        if i==1 and j==1:
            tp += 1
        if i==1 and j==0:
            fp += 1
        if i==0 and j==1:
            fn += 1
        if i==0 and j==0:
            tn += 1

    print('tp is',str(tp))
    print('fp is', str(fp))
    print('tn is', str(tn))
    print('fn is', str(fn))
    print('prediction is ',str(float(tp/(tp+fp))))
    print('recall is ',str(float(tp/(tp+fn))))



def spilt_right_wrong(Y, y, result):
    print(result)
    for f, i in zip(result, range(len(result))):
        if f:
            right_set.append([Y[i], y[i]])
        else:
            wrong_set.append([Y[i], y[i]])


    rightsize = 0
    wrongsize = 0
    for r in right_set:
        if r[0] == 1:
            rightsize += 1
        else:
            wrongsize += 1
    print('in rightset,predict counterfactual size is ', str(rightsize), 'predict uncounterfactual size is ',
          str(wrongsize))
    print('rate is ', float(rightsize / len(right_set)))

    rightsize = 0
    wrongsize = 0
    for r in wrong_set:
        if r[0] == 1:
            rightsize += 1
        else:
            wrongsize += 1
    print('in wrongset,predict counterfactual size is ', str(rightsize), 'predict uncounterfactual size is ',
          str(wrongsize))
    print('rate is ', float(rightsize / len(wrong_set)))




def train(x_train, y_train, feature_size):
    train_data_size = x_train.shape[0]
    print('train set size is ',str(train_data_size))
    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((feature_size,))
    mu2 = np.zeros((feature_size,))
    for i in range(train_data_size):
        if y_train[i] == 1:
            mu1 += x_train[i]
            cnt1 = cnt1 + 1
        else:
            mu2 = mu2 + x_train[i]
            cnt2 = cnt2 + 1
    mu1 = mu1 / cnt1
    mu2 = mu2 / cnt2

    sigma1 = np.zeros((feature_size, feature_size))
    sigma2 = np.zeros((feature_size, feature_size))
    for i in range(train_data_size):
        if y_train[i] == 1:
            sigma1 = sigma1 + np.dot(np.transpose([x_train[i] - mu1]), [x_train[i] - mu1])
        else:
            sigma2 = sigma2 + np.dot(np.transpose([x_train[i] - mu2]), [x_train[i] - mu2])

    sigma1 = sigma1 / cnt1
    sigma2 = sigma2 / cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

    n1 = cnt1
    n2 = cnt2

    return mu1, mu2, shared_sigma, n1, n2

def dis_false_answer(y,Y,parses):
    for i,j,k in zip(y,Y,parses):
        if i <0.5 and j ==1:
            print(str(i),'   ',str(j),'  ',k)

    for i,j,k in zip(y,Y,parses):
        if i >0.5 and j ==0:
            print(str(i),'   ',str(j),'  ',k)

def get_test_data(testsenvec,X,mu1,mu2,shared_sigma,N1,N2,feature_size):
    sigma_inv = inv(shared_sigma)  # 相同的协方差即可
    w = np.dot((mu1 - mu2), sigma_inv)
    X_t = X.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    print('testpred display')
    # dis_false_answer(y,Y,parses_train2)r

    y_ = np.around(y)  # 返回四舍五入后的值


    n_row = 0

    needlist = []
    for senid,pred in zip(testsenvec,y_):
        needlist.append([])
        needlist[n_row].append(senid.tags[0])
        needlist[n_row].append(pred)
        n_row = n_row + 1

    df2 = pd.DataFrame(needlist)
    df2.rename(columns={0: 'sentenceID', 1: 'pred_label'}, inplace=True)
    print(df2)
    df2.to_csv(os.path.join('subtask1_pred.csv'), index=False)

    return



def execute(sen,tsen,set_size):
    for s in set_size:
        row_num = len(sen)
        trow_num = len(tsen)
        sentence_vec,tsen_vec = get_vec(sen,tsen,s)
        print(len(sentence_vec),len(tsen_vec))
        #sentence_vec,tsen_vec = get_from_disk(parses,s)

        df = pd.DataFrame(sentence_vec.reshape(row_num,s))
        #testdf = pd.DataFrame(tsen_vec.reshape(trow_num,s))

        x_train = df.values
        #x_test = testdf.values
        y_train = dataprocess_y(data_x).values

        vaild_set_percetange = 0
        X_train, Y_train, X_valid, Y_valid, parses_train2 = split_valid_set(x_train, y_train, vaild_set_percetange, parses)
        mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train,s)
        get_test_data(tsen,tsen_vec,mu1, mu2, shared_sigma, N1, N2, s)
        #valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2, s)
        valid_on_train(X_train, Y_train, mu1, mu2, shared_sigma, N1, N2, s, parses_train2)

def to_text3(testdata):
    output = open('subtask1testoutput.txt', 'w+', encoding='utf-8')
    for l in testdata:
        output.write(l)
        output.write('\n')
    output.close()



n_row = 0
text = pd.read_csv('gd_output23000.csv')
tdata = pd.read_csv('subtask1_test.csv')


data_x = text[['sentenceID','gold_label','sentence']]
tdata_x = tdata[['sentenceID','sentence']]


sentenses = data_x['sentence'].tolist()
trainid = data_x['sentenceID'].tolist()
tsentences = tdata_x['sentence'].tolist()
testid = tdata_x['sentenceID'].tolist()

sentenses = list(_flatten(del_nonuse(sentenses)))
tsentences = list(_flatten(del_nonuse(tsentences)))

parses = sentenses
to_txt2(parses)
parses = np.array(parses)

size_list = [91]

trainlabelized,testlabelized = get_x_data(sentenses,trainid,tsentences,testid)
execute(trainlabelized,testlabelized,size_list)


