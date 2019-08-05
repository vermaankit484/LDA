import nltk
import numpy as np
from nltk.corpus import stopwords
set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
print("Enter the number of topics")
tp=int(input())
tparray=[]
for i in range(tp):
    tparray.append(i)
#eg=pd.read_csv("abcnews-date-text.csv", header =[1, 2])
#print(ds)
stop_words = set(stopwords.words('english'))#Loading English Stopwords
file1 = open("dataset.csv", "r")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
cnt=0
for r in words:
    r=r.lower()
    if(r[:3] == "doc"):
        cnt+=1
list=[[] for i in range(cnt)]#Creating 2d empty list
doctop=[[] for q in range(cnt)]
i=-1
for r in words:
    if not r in stop_words:
        r=r.lower()#Lower Case the Words
        r = ps.stem(r)#Stemming the words
        if (r[:3] == "doc"):
            i+=1
        if(r[:3] =="doc"):
            p=r.find(',')
            r=r[p+1:]
        list[i].append(r)#Appending at the end of list
pp=0
for i in range(len(list)):
    for j in range(len(list[i])):
        pp+=1
        x = np.random.randint(0, tp)#Used for random assignment of topics
        doctop[i].append(x)
m_set=set([])#Set DS used for getting dis0tinct values
print("2d List representing random assignment of topics")
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        m_set.add(list[i][j])#Add each word in the set
        print(doctop[i][j], end=" ")
    print("\n")
arr=np.full((cnt, tp), 0)#Used to create 2d numpy array
#arr array will correspond to frequency of each topic in each document, rows-documents, cols-topics
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        arr[i][doctop[i][j]]+=1
print("2d numpy array representing frequency of topics in documents")
print(arr)
tpword=np.full((tp, len(m_set)), 0)#2d numpy array which will correspond to frequency of each word correspoding to each topic
Dict={}#Dictionary used to map words or to create associative arrays
c=0
for val in m_set:
    Dict[val]=c#Adding a word in Dictionary
    c+=1
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        tpword[doctop[i][j]][Dict[list[i][j]]]+=1#Used to increment frequency count
        #rows represent topic, cols represents words
print("2d numpy array representing frequency of words in each topic")
print(tpword)
"""print(np.sum(tpword))
print(len(m_set))
print(pp)"""
#prarr=np.full((cnt, tp), 0)
prarr=[[] for q in range(cnt)]
for i in range(len(arr)):
    s=0
    for j in range(len(arr[i])):
        s+=arr[i][j]
    for j in range(len(arr[i])):
        v=arr[i][j]/s
        #prarr[i][j]=v
        if v==0:
            v=1/(s+(len(arr[i])+1) )
        prarr[i].append(v)
"""for i in range(len(prarr)):
    for j in range(len(prarr[i])):
        print(prarr[i][j],end=' ')
    print('\n')"""
prtpwrd=[[] for q in range(tp) ]
for i in range(len(tpword)):
    s=0
    for j in range(len(tpword[i])):
        s+=tpword[i][j]
    for j in range(len(tpword[i])):
        v=tpword[i][j]/s
        if v==0:
            v=1/(s+(len(tpword[i]) +1))
        prtpwrd[i].append(v)
"""print('\n\n')
for i in range(len(prtpwrd)):
    for j in range(len(prtpwrd[i])):
        print(prtpwrd[i][j], end=' ')
    print('\n')"""
print("Before LDA")
for i in range(len(prarr)) :
    prarr[i].append('B')
#print(prarr)
for i in range(len(prarr)) :
    print("Doc", end="")
    print(i, end=" ")
    for j in range(len(prarr[i])) :
        if(j!=len(prarr[i])-1) :
            print(prarr[i][j],end=' ')
    print('\n')
for i in range(len(prarr)) :
    del(prarr[i][len(prarr[i])-1])
file1=open("grph1.txt", "a")
for i in range(len(prarr)) :
    file1.write('{}'. format(prarr[i]))
for p in range(200):
    chs=[None]*tp
    for i in range(len(list)):
        for j in range(len(list[i])):
            for k in range(tp):
                chs[k]=prarr[i][k]
                chs[k]=chs[k]*prtpwrd[k][Dict[list[i][j]]]
            sum=0
            for k in range(tp):
                sum+=chs[k]
            for k in range(tp):
                chs[k]=chs[k]/sum
            #print(chs)
            pre=doctop[i][j]
            post=int(np.random.choice(tparray, 1, chs))
            doctop[i][j] = post
            arr[i][pre]=max(0, arr[i][pre]-1)
            arr[i][post]+=1
            tpword[pre][Dict[list[i][j]]]=max(0, tpword[pre][Dict[list[i][j]]]-1)
            #print(Dict[list[i][j]], post)
            tpword[post][Dict[list[i][j]]]+=1
            s=0
            for k in range(len(arr[i])) :
                s+=arr[i][k]
            for k in range(len(arr[i])) :
                v=arr[i][k]/s
                if v == 0:
                    v = 1 / (s + (len(arr[i]) + 1))
                prarr[i][k]=v
            s=0
            for k in range(len(prtpwrd[pre])) :
                s+=tpword[pre][k]
            for k in range(len(prtpwrd[pre])):
                v=tpword[pre][k]/s
                if v == 0:
                    v = 1 / (s + (len(prtpwrd[pre]) + 1))
                prtpwrd[pre][k]=v
            s = 0
            for k in range(len(prtpwrd[post])):
                s += tpword[post][k]
            for k in range(len(prtpwrd[post])):
                v = tpword[post][k] / s
                if v == 0:
                    v = 1 / (s + (len(prtpwrd[post]) + 1))
                prtpwrd[post][k] = v
for i in range(len(prarr)) :
    prarr[i].append('A')
print("After LDA")
for i in range(len(prarr)) :
    print("Doc", end = "")
    print(i, end = " ")
    for j in range(len(prarr[i])) :
        if(j!=len(prarr[i])-1) :
            print(prarr[i][j],end=' ')
    print('\n')
for i in range(len(prarr)) :
    del(prarr[i][len(prarr[i])-1])
#print(prarr)
