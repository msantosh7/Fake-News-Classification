import pandas as pd 
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

TFIDF = TfidfVectorizer()

def calc_similarity(row): 
    row = row[0:2]
    tfidf_matrix = TFIDF.fit_transform(row)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(cosine_sim[0][1])
    return cosine_sim[0][1]


# read data
x = pd.read_csv('similarities.csv')
y = pd.read_csv('labels.csv')
x = np.array(x['sim']).reshape(-1, 1)
y = y['label']

# split into testing and training 
train_x, test_x, train_y, test_y = train_test_split(x,y , test_size=0.3, random_state=1)

# balance data 
sm = SMOTE(random_state=1)
train_x_r, train_y_r = sm.fit_resample(train_x, train_y.ravel()) 

# ------------------------------------------------ # 
# train multinomial linear regression model 
# 66% accuracy, 71% weighted f1 
# ------------------------------------------------ # 
# clf = LogisticRegression(solver='lbfgs',multi_class='multinomial')
# clf.fit(train_x_r, train_y_r)
# clf_y_pred = clf.predict(test_x)
# # print results 
# print("Accuracy:")
# print(metrics.accuracy_score(test_y, clf_y_pred)*100)
# print(metrics.confusion_matrix(test_y, clf_y_pred))
# print(metrics.classification_report(test_y, clf_y_pred))


# ------------------------------------------------ # 
# train SVC model 
# 74% accuracy, 74% weighted f1 
# ------------------------------------------------ # 
clf = LinearSVC()
clf.fit(train_x_r, train_y_r)
clf_y_pred = clf.predict(test_x)
# print results 
print("Accuracy:")
print(metrics.accuracy_score(test_y, clf_y_pred)*100)
print(metrics.confusion_matrix(test_y, clf_y_pred))
print(metrics.classification_report(test_y, clf_y_pred))


### NOTE I don't think random forest or Naive Bayes are good in general, since we only have 1 feature ### 

# ------------------------------------------------ # 
# train random forest model 
# 58% accuracy, 66% weighted f1 
# ------------------------------------------------ # 
# rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
# rfc.fit(train_x_r, train_y_r)
# rfc_pred = rfc.predict(test_x)
# # print results 
# print("Accuracy:")
# print(metrics.accuracy_score(test_y, rfc_pred)*100)
# print(metrics.confusion_matrix(test_y, rfc_pred))
# print(metrics.classification_report(test_y, rfc_pred))



# ------------------------------------------------ # 
# train naive bayes model 
# 68% accuracy, 71% weighted f1 
# ------------------------------------------------ # 
# nb = GaussianNB()
# nb.fit(train_x_r, train_y_r)
# nb_pred = nb.predict(test_x)
# # print results 
# print("Accuracy:")
# print(metrics.accuracy_score(test_y, nb_pred)*100)
# print(metrics.confusion_matrix(test_y, nb_pred))
# print(metrics.classification_report(test_y, nb_pred))


# ------------------------------------------------ # 
# train sgd model 
# 69% accuracy, 73% weighted f1 
# ------------------------------------------------ # 
# sgd = SGDClassifier(loss="log_loss", penalty="l2")
# sgd.fit(train_x_r, train_y_r)
# sgd_pred = sgd.predict(test_x)
# # print results 
# print("Accuracy:")
# print(metrics.accuracy_score(test_y, sgd_pred)*100)
# print(metrics.confusion_matrix(test_y, sgd_pred))
# print(metrics.classification_report(test_y, sgd_pred))




# #test data
test = pd.read_csv('cleaned_test.csv')
test.drop(columns=['Unnamed: 0'],inplace=True)

test_without_id = test.drop(columns='id')

# calculate similarity between titles 
test_without_id['sim']=test_without_id.apply(lambda row: calc_similarity(row), axis=1)

x_test = np.array(test_without_id['sim']).reshape(-1,1)
y_pred_clf = clf.predict(x_test) #predicting test data using Logistic Regression

#mapping back 0, 1, 2 with 'unrelated', 'disagreed', 'agreed' respectively
y_pred_clf1 = np.where(y_pred_clf == 0, 'unrelated', np.where(y_pred_clf == 1, 'disagreed', np.where(y_pred_clf == 2, 'agreed',y_pred_clf)))
print(y_pred_clf1)

clf_label = pd.DataFrame(y_pred_clf1, columns=['label'])
submission = pd.concat([test, clf_label], axis=1)
submission.drop(columns=['title1_en','title2_en'], inplace=True)

#write to submission.csv
submission.to_csv('submission.csv',index=False)