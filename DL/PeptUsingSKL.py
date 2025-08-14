from sklearn.linear_model import Perceptron

from sklearn.datasets import make_classification 

from sklearn.model_selection import train_test_split


#creating & splitiing data
x,y = make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=42) #x-> feature, y-> label 

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) Here, 30% of data will go to the test set. ex: output(750,10) 10 means-> number of features in your dataset
'''
#function used to generate sythetic(fake) classification datasets for testing and practicing ml models.

#random state: seed for reproducibility
#feature : number o column
#sample: number of rows
#classes: number of classes(output categories)


#hyperparameter of perceptron

#initlize the perceptron

clf = Perceptron(
    max_iter = 1000, # Max number of epchos
    eta0= 0.1, #learning rate 
    random_state = 42, # for reproducbiability
    tol=1e-3, #stop early if improvement is smaller than this 
    shuffle = True #shuffle data each epoch
)


clf.fit(x_train,y_train)

accuracy  = clf.score(x_test,y_test)

print(f"The Accuracy is {accuracy}")