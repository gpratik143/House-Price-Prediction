import numpy as np
from sklearn.datasets import fetch_openml
dataset=fetch_openml('mnist_784')
data=dataset['data']
type(data)
tar=dataset['target']
type(tar)
tar=tar.apply(int)
tar[:1]
type(tar[:1][0])

import matplotlib.pyplot as plt

for i,j in enumerate(np.array(data.head(100))):    # i store the index value of row staring from 0 to 100 and j stores the data present in corresponding rows starting from row[0] data to ro[100] data
    plt.subplot(10,10,i+1)  #will create a grid of 10X10
    plt.axis('off')
    plt.imshow(j.reshape(28,28),cmap='binary')
    
plt.show()
data.shape
#traning the data by seperating first 60000 train data and last 10000 test data

X_train=data[:60000]
X_test=data[60000:]

y_train=tar[:60000]
y_test=tar[60000:]

#For classification the class -modify the target value only by providing it as low value(false) if target is not same and if target is obtained then high value(true)
  
y_train_5=(y_train==5)   # y_train_5 contains 60000 rows and 1 column and value stored in each rows is boolean value i.e either true or False
y_train_5.head(5)

y_test_5=(y_test==5)
y_test_5.head(5)
# the above 4 line is done to do Binary Classification

#No change is done in X_train and X_test data

from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(random_state=42)


# Train the model
sgd.fit(X_train,y_train_5)

#prediction

pred_5=sgd.predict(X_test)

# ACCURACY CHECK:
    
#->METHOD 1:
(y_test_5==pred_5).sum()/100     ##((pred_x==y_train_5).sum()/10000)*100  

pred_x=sgd.predict(X_train)
(pred_x==y_train_5).sum()/600     #((pred_x==y_train_5).sum()/60000)*100    
#->METHOD 2:
from sklearn.metrics import accuracy_score
accuracy_pred_5=accuracy_score(y_test_5,pred_5)
print(accuracy_pred_5)  #0.9492 <=>94.92%


# GRAY AREA in model : it is the percentage of error i.e here 6% having accuracy of 94%
a=[]
for i,j,k in zip(range(len(y_test)),pred_5,y_test_5):
    if(j==True and k==False):
        #print(i)
        a.append(i)
    

print(a[:5])  #giving first 5 value  where pred_5 == true and 
len(a)
print(type(y_test))
print(y_test[98:99])
plt.imshow((np.array(X_test[98:99]).reshape(28,28)),cmap='binary')
print(y_test[112:113])
plt.imshow(np.array(X_test[112:113]).reshape(28,28))
print(y_test[148:149])
plt.imshow(np.array(X_test[148:149]).reshape(28,28))
print(y_test[170:171])
plt.imshow(np.array(X_test[170:171]).reshape(28,28))
print(y_test[192:193])
plt.imshow(np.array(X_test[192:193]).reshape(28,28))

#Home assignment
#print first 100 digit where prediction saying 5 but ground truth tell not 5
#print first 100 digit where prediction not saying 5 but ground truth tell  5
#do using subplot

#assignment question 1 (soln.):
a1=[]
for i,j,k in zip(range(len(y_test)),pred_5,y_test_5):
    if(j==True and k==False):
        #print(i)
        a1.append(i)
for i,j in enumerate(np.array(a1[:100])):
    plt.subplot(10,10,i+1)
    plt.axis('off')
    plt.imshow((np.array(X_test[j:j+1])).reshape(28,28),cmap='binary')
plt.show()

#assignment question 2(soln.):

a2=[]
for i,j,k in zip(range(len(y_test)),pred_5,y_test_5):
    if(j==False and k==True):
        #print(i)
        a2.append(i)
for i,j in enumerate(np.array(a2[:100])):
    plt.subplot(10,10,i+1)
    plt.axis('off')
    plt.imshow((np.array(X_test[j:j+1])).reshape(28,28),cmap='binary')
plt.show()