import numpy as np
import time
import string
import math
import sys
import random as rand
alphabet = "abcdefghijklmnopqrstuvwxyz"
random = rand.getrandbits(1)
syn0 = np.loadtxt('result1.txt')
syn1 = np.loadtxt('result2.txt')
syn2 = np.loadtxt('result3.txt')
syn3 = np.loadtxt('result4.txt')


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

while 1:
     
    i=0
    wordpos=[0]*30
    data = np.zeros(27*30)
    word = input('word ')
    for letter in word:
        wordpos[i] = alphabet.find(letter)+1
        i+=1
    a = 0
    for i in wordpos:
        data[i+a]=1
        a+=27
    X = np.array([data])

    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    l4 = nonlin(np.dot(l3, syn3))
    if l4 < 0.5:

        print("i am " + str(((1-(np.round(l4,3)))*100)) + "% sure this word is english")
    else:
        print("i am " + str(((np.round(l4,3))*100)) + "% sure this word is dutch")
