
# coding: utf-8

# In[22]:


import random,math,copy,time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


# In[23]:

     
#Voice
#df=pd.read_csv("/Users/naman/Downloads/Voice PD Dataset.csv",dtype='category')
#df=pd.read_csv("/Users/naman/Desktop/wndOutputFeatureMatrix.csv",dtype='category')
#df=pd.read_csv("/Users/naman/Desktop/Lung Dataset Output_final_segmented/WndDataset/wndOutputFeatureMatrix.csv",dtype='category')
#df=pd.read_csv("/Users/naman/Desktop/LungsDataset.csv",dtype='category')
df=pd.read_csv("/Users/naman/Desktop/?EEG/EEG.csv")

df=shuffle(df)


features=[]

#52

for k in range(177):
	features.append(str(k))


d=len(features)   # Flock (population) size
lt=len(features)
y=df['177']
x=df[df.columns[0:177]]

#df2=pd.read_csv("/Users/naman/Downloads/test_data.csv",dtype='category')

#y2=df2['27']
#x2=df2[df.columns[0:27]]

#df = shuffle(df)


# In[24]:


wf=0.8# wf is used to control the importance of classification accuracy and number of selected features.

dim=len(features)
Max_iter=100
lb=0
ub=1
 
SearchAgents_no=12
#initialize alpha, beta, and delta_pos
Alpha_pos=np.zeros(dim)
Alpha_score=float(0)
Beta_pos=np.zeros(dim)
Beta_score=float(0)
    
Delta_pos=np.zeros(dim)
Delta_score=float(0)
    
#Initialize the positions of search agents
#transform it into 0 and 1 if greater than 0.5
Positions=np.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
Convergence_curve=np.zeros(Max_iter)


# In[25]:



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

kCross=10


neigh = KNeighborsClassifier(n_neighbors=6)
scores = cross_val_score(neigh, x, y, cv=kCross, scoring='accuracy')
print(scores)
print('KNN: ',scores.mean())


'''
neigh.fit(x_train, y_train)
neigh.predict(x_test)
nacc=neigh.score(x_test,y_test)
'''


rforest = RandomForestClassifier(n_estimators=500)
rscores = cross_val_score(rforest, x, y, cv=kCross, scoring='accuracy')
print(rscores)
print('RF: ',rscores.mean())

'''
rforest.fit(x_train, y_train)
rforest.predict(x_test)
rfacc=rforest.score(x_test,y_test)
'''



dtree = tree.DecisionTreeClassifier(max_depth=30, min_samples_split=20)
dscores = cross_val_score(dtree, x, y, cv=kCross, scoring='accuracy')
print(dscores)
print('DT: ',dscores.mean())


'''
dtree.fit(x_train, y_train)
dtree.predict(x_test)
dtacc=dtree.score(x_test,y_test)
'''

SVM = svm.SVC(kernel='linear', C=1.0, gamma=1)
#SVM = svm.SVC(kernel='poly', degree=2)
#SVM = svm.SVC(kernel='rbf')

sscores = cross_val_score(SVM, x, y, cv=kCross, scoring='accuracy')
print(sscores)
print('SVM: ',sscores.mean())


'''
SVM.fit(x_train, y_train)
y_pred = SVM.predict(x_test)
'''


'''
print ('Classification accuracy before feature selection:')
print('K Neighbors Classifier: ',nacc)
print('Random forest Classifier: ',rfacc)
print('Decision tree Classifier: ',dtacc)
'''
#print('SVM Classifier: ',metrics.accuracy_score(y_test, y_pred))


model=RandomForestClassifier()
model.fit(x_train,y_train)
im=(model.feature_importances_)
im


# In[26]:


imo=pd.Series()
for i in range(d):
    imo= imo.set_value(features[i],im[i])
print(imo)


# In[27]:


# this defines the fitness of each crow by calculating their accuracy and no features selected using the below formula
def fit(df2):      # fitness function whose 1 parameter takes the crows and their position as input
    global sel
    sel=[]
    q=0
    imp=0
    a=(df2)==0.0
    if a.all()==True:
        df2[:]=1.0
    for p in range(len(features)):
        if (df2[p]==1.0):
            sel.insert(q,p)
            temp=imo[p]
            imp=imp+temp
        q=q+1
    lf=len(sel)
    divide=lf/lt
    fnt=imp+wf*(1-float(lf/lt))     # fitness formula
    return fnt


# In[28]:


time1=time.time()

for l in range(0,Max_iter):

    for i in range(0,SearchAgents_no):
            
            # Calculate objective function for each search agent
        fitness=fit(Positions[i,:])
            # Update Alpha, Beta, and Delta
        if fitness>Alpha_score :
            Alpha_score=fitness; # Update alpha
            Alpha_pos=np.array(Positions[i,:])
        
        if (fitness<Alpha_score and fitness>Beta_score ):
            Beta_score=fitness  # Update beta
            Beta_pos=np.array(Positions[i,:])
            
            
        if (fitness<Alpha_score and fitness<Beta_score and fitness>Delta_score): 
            Delta_score=fitness # Update delta
            Delta_pos=np.array(Positions[i,:])
            
        
        
        
    a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
    for i in range(0,SearchAgents_no):
        for j in range (0,dim):
                           
            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]
                
            A1=2*a*r1-a; # Equation (3.3)
            C1=2*r2; # Equation (3.4)

            D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
            X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
            
            r1=random.random()
            r2=random.random()
                
            A2=2*a*r1-a; # Equation (3.3)
            C2=2*r2; # Equation (3.4)
                
            D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
            X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
            r1=random.random()
            r2=random.random() 
                
            A3=2*a*r1-a; # Equation (3.3)
            C3=2*r2; # Equation (3.4)
                
            D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
            X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3
            
            X=(X1+X2+X3)/3
            sig_X= 1/( 1 + math.exp(( 10*(X-0.5 ) ) ))
            Positions[i,j]= float(sig_X) 

    Positions=(Positions>=random.random()).astype(float)
        
    Convergence_curve[l]=Alpha_score;

    if (l%1==0):
            print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
time2=time.time()
tottime=(time2-time1)
print(tottime)
  


# In[29]:


Alpha_pos


# In[30]:


# here the accuracy of each crows are calculated.

q=0

select=[]
for i in range(len(Alpha_pos)):
    if Alpha_pos[i]==1:
        select.insert(q,features[i])
        q=q+1


print('No. of features selected: ',len(select))

print(select)

X=x[select]
Y=y



print ('Classification accuracy after feature selection:')
Kscores = cross_val_score(neigh, X, Y, cv=kCross, scoring='accuracy')
print(Kscores)
print('KNN: ',Kscores.mean())


Rscores = cross_val_score(rforest, X, Y, cv=kCross, scoring='accuracy')
print(Rscores)
print('RF: ',Rscores.mean())

Dscores = cross_val_score(dtree, X, Y, cv=kCross, scoring='accuracy')
print(Dscores)
print('DT: ',Dscores.mean())

Sscores = cross_val_score(SVM, X, Y, cv=kCross, scoring='accuracy')
print(Sscores)
print('SVM: ',Sscores.mean())























'''
neigh.fit(x_train[select], y_train)
neigh.predict(x_test[select])
nacc1=neigh.score(x_test[select],y_test) 
    
rforest.fit(x_train[select], y_train)
rforest.predict(x_test[select])
rfacc1=rforest.score(x_test[select],y_test) 

dtree.fit(x_train[select], y_train)
dtree.predict(x_test[select])
dtacc1=dtree.score(x_test[select],y_test)

print (nacc1, rfacc1, dtacc1) # the crow with maximum accuracy is printed
'''

# In[31]:


plt.plot(Convergence_curve)
plt.xlabel('No of iterations')
plt.ylabel('Fitness value')
plt.show()

