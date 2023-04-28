import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleaned_train.csv')

train.drop(columns=['Unnamed: 0'], inplace=True)
# train = train.head()

# calculate cosine similarity between two word vectors 
def calc_similarity(row): 
    row = row[0:2]
    tfidf_matrix = TFIDF.fit_transform(row)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(cosine_sim[0][1])
    return cosine_sim[0][1]


TFIDF = TfidfVectorizer()

# calculate similarity between titles 
train['sim']=train.apply(lambda row: calc_similarity(row), axis=1) 

# get x, y data 
# x = np.array(train['sim']).reshape(-1, 1)
x = train['sim']
y = train['label']

x.to_csv('similarities.csv')
y.to_csv('labels.csv')
