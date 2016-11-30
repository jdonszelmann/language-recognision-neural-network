import numpy as np
import matplotlib.pyplot as plt
import random as rand
alpha,hiddenNeurons = (0.1,100)

lang1 = open('English.txt')
lang2 = open('Dutch.txt')
lang1lines = lang1.readlines()
lang2lines = lang2.readlines()
alphabet = "abcdefghijklmnopqrstuvwxyz"

training = int(input('cycles '))

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def getWords(lines):
    done = False
    while done == False:
        try:
            randint = rand.randint(10,len(lines))
            line = lines[randint]
            done = True
        except:
            pass

    if '/' in line:
            line = line.split('/', 1)[0]
    line = line.strip()
    wordpos=[0]*30
    i=0
    data = np.zeros(27*30)
    for letter in line:
        if letter in alphabet:
            try:
                wordpos[i] = alphabet.find(letter)+1
                i+=1
            except:
                pass
    a = 0
    for i in wordpos:
        data[i+a]=1
        a+=27
    X = np.array([data])
    return X,line

for iterations in range(1,training):

    random = rand.getrandbits(1)
    if random == 0:
        #y = np.array([[1]]).T #ENGELS
        y = np.array([[0]]).T #ENGELS
        X,word = getWords(lang1lines)
        color = 'bo'
    if random == 1:
        y = np.array([[1]]).T #NEDERLANDS
        X,word = getWords(lang2lines)
        color = 'ro'

    if iterations == 1:
        syn0 = 2*np.random.random((X.shape[1],hiddenNeurons)) - 1
        syn1 = 2*np.random.random((hiddenNeurons,hiddenNeurons)) - 1
        syn2 = 2*np.random.random((hiddenNeurons,hiddenNeurons)) - 1
        syn3 = 2*np.random.random((hiddenNeurons,y.shape[1])) - 1
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    l4 = nonlin(np.dot(l3, syn3))
    
    l4_error = l4 - y
    l4_delta = l4_error * nonlin(l4,True)

    l3_error = l4_delta.dot(syn3.T)
    l3_delta = l3_error * nonlin(l3,True)

    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(l2,True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,True)
    
    syn3 -= (alpha * l3.T.dot(l4_delta))
    syn2 -= (alpha * l2.T.dot(l3_delta))
    syn1 -= (alpha * l1.T.dot(l2_delta)) 
    syn0 -= (alpha * l0.T.dot(l1_delta))

    if iterations%10000 == 0:
        plt.plot(iterations,l4,color)

    if iterations%100000 == 0:
        print(iterations,word,l4_error)
    if iterations % 1000000 == 0:
        np.savetxt('result1.txt',syn0)
        np.savetxt('result2.txt',syn1)
        np.savetxt('result3.txt',syn2)
        np.savetxt('result4.txt',syn3)

print(l4)
np.savetxt('result1.txt',syn0)
np.savetxt('result2.txt',syn1)
np.savetxt('result3.txt',syn2)
np.savetxt('result4.txt',syn3)
plt.show()