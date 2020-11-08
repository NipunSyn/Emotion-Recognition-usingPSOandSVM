
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn import tree

df = pd.read_csv("C:\\Users\\Documents\\Emotions Edited.csv") ## Loading the dataset

df.drop('Emotion',axis = 1, inplace = True)
col = df.columns[0:676]
df.drop(col,inplace = True, axis =1 )

x = df.drop('Target',axis =1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.10,random_state =42) 
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.20,random_state = 42)

## PARTICLE SWARM OPTIMIZATION
#Initialising Parameters
w=0.8
c1=2
c2=2
popsize=30
itr=500
gbestfitness= 0
gbest=np.random.uniform(0,0,len(x.columns))



class Particle:
    def __init__(self):
        self.pos=np.random.uniform(0,1,len(x.columns)) ## initializing a random start position
        self.vel=np.random.uniform(0,1,len(x.columns)) ## initializing random velocity
    
        self.bits=np.random.uniform(0,0,len(x.columns))
        for i in range(0,len(x.columns)):
            self.bits[i] = self.pos[i]
            if self.bits[i]>0.94:
                self.bits[i] = 1 
            else:
                self.bits[i] = 0
        self.bits = self.bits.astype(int)
    
        self.fitness = 0 
        self.lbfitness = 0
        self.lbest = self.pos

## constructing the particle swarm 
particles = [] 
for i in range(popsize):
    particles.append(Particle())
    
gbest_accuracy = [] 
for j in range(0,itr):
    for i in range(0,popsize):
        lst = [] 
        for k in range(0,len(x.columns)):
            if particles[i].bits[k]==0:
                lst.append(k)
        xPSO = X_train.drop(X_train.columns[lst],axis=1) ## selected feature set
        
        clf = svm.SVC()## linear kernel     
        ##training the model
        try:
            clf.fit(xPSO,y_train)
        
        #predict the responses for test dataset
            xPSO_val = X_val.drop(X_val.columns[lst],axis=1)
        
            y_pred = clf.predict(xPSO_val)
            particles[i].fitness = metrics.accuracy_score(y_val,y_pred)
        except:
            particles[i].fitness = 0
        
        if particles[i].fitness>particles[i].lbfitness: 
            particles[i].lbfitness = particles[i].fitness
            particles[i].lbest = particles[i].pos
        
        if particles[i].lbfitness > gbestfitness:
            gbestfitness = particles[i].lbfitness
            gbest = particles[i].lbest
    
    gbest_accuracy.append(gbestfitness)
    
    for i in range(0,popsize): 
        particles[i].vel = w*particles[i].vel + c1*random.random()*(particles[i].lbest - particles[i].pos)+ c2*random.random()*(gbest - particles[i].pos)
        particles[i].pos = particles[i].pos + particles[i].vel

gbbits = np.random.uniform(0,0,len(x.columns))

for i in range(0,len(x.columns)):
    gbbits[i] = gbest[i]
    if gbbits[i] > 0.94: 
        gbbits[i] = 1
    else: 
        gbbits[i] = 0 
gbbits = gbbits.astype(int)

lst1=[]
for i in range(0,len(x.columns)):
    if gbbits[i]==0:
        lst1.append(i)
        
gbxPSO = X_train.drop(X_train.columns[lst1],axis=1)

## TESTING

clf =  svm.SVC(kernel = 'linear') # Linear Kernel
##clf = tree.DecisionTreeClassifier()
clf.fit(gbxPSO,y_train)

gbxPSO_test = X_test.drop(X_test.columns[lst1],axis=1) 
y_pred = clf.predict(gbxPSO_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(100*accuracy)