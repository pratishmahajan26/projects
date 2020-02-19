# import the libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

# cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
corpus = []
for i in range(0,dataset.shape[0]):
    review = dataset['Review'][i]
    # first remove everything except alphabets
    review = re.sub('[^a-zA-Z]',' ',review)
    # convert all to lower case
    review = review.lower()
    # now convert all the words into a list
    review = review.split()
    # removing the stopwords and stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)
    
# now we will create bag of words matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()  # to convert it into a matrix
y = dataset.iloc[:,1].values

# training the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# appying naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# prediction
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)
 # k-fold validation
from sklearn.model_selection import cross_val_score
validate = cross_val_score(estimator = classifier, X = X_train, y = y_train,cv = 10)
validate.mean()
    

