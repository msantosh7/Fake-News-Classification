import pandas as pd 
import string
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()

def remove_punctuation(text): 
    text_list = [char for char in text if char not in PUNCT]
    clean = ''.join(text_list)
    return clean

def split_text(text): 
    return text.split(' ')

def remove_stopwords(text): 
    text_words = set(text)
    text_without_stopwords = text_words.difference(STOPWORDS)
    return list(text_without_stopwords)

def remove_nums(text): 
    nums = set(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])
    text = list(set(text).difference(nums))
    ret = [] 
    for word in text: 
        try: 
            int(word)
        except: 
            ret.append(word)
    return ret

def lemmatize(text): 
    text = [LEMMATIZER.lemmatize(x) for x in text if x]
    return text

with open('./english', 'r') as f: 
    STOPWORDS = f.read().splitlines()

PUNCT = string.punctuation
# print(PUNCT.replace('\'', ''))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop unused columns
train.drop(columns = ['id', 'tid1', 'tid2'], inplace = True) 
test.drop(columns=['tid1','tid2'], inplace=True)

train.label = train.label.astype(str)
train.label = train.label.str.strip() 

# replace labels with numerics 
replace = {'unrelated': '0', 'disagreed': '1', 'agreed': '2'} 
train['label'] = train['label'].map(replace)

# convert titles to lowercase (train)
train['title1_en'] = train['title1_en'].apply(lambda x: x.lower())
train['title2_en'] = train['title2_en'].apply(lambda x: x.lower())
# convert titles to lowercase (test)
test['title1_en'] = test['title1_en'].apply(lambda x: x.lower())
test['title2_en'] = test['title2_en'].apply(lambda x: x.lower())


# remove punctuation from titles (train)
train['title1_en'] = train['title1_en'].apply(remove_punctuation)
train['title2_en'] = train['title2_en'].apply(remove_punctuation)
# remove punctuation from titles (test)
test['title1_en'] = test['title1_en'].apply(remove_punctuation)
test['title2_en'] = test['title2_en'].apply(remove_punctuation)


train['title1_en'] = train['title1_en'].apply(split_text)
train['title2_en'] = train['title2_en'].apply(split_text)

test['title1_en'] = test['title1_en'].apply(split_text)
test['title2_en'] = test['title2_en'].apply(split_text)

# remove common words from titles (train)
train['title1_en'] = train['title1_en'].apply(remove_stopwords)
train['title2_en'] = train['title2_en'].apply(remove_stopwords)
# remove common words from titles(test)
test['title1_en'] = test['title1_en'].apply(remove_stopwords)
test['title2_en'] = test['title2_en'].apply(remove_stopwords)

# remove numbers from titles (train)
train['title1_en'] = train['title1_en'].apply(remove_nums)
train['title2_en'] = train['title2_en'].apply(remove_nums)
# remove numbers from titles(test)
test['title1_en'] = test['title1_en'].apply(remove_nums)
test['title2_en'] = test['title2_en'].apply(remove_nums)

# including lemmatization (train)
train['title1_en'] = train['title1_en'].apply(lemmatize)
train['title2_en'] = train['title2_en'].apply(lemmatize)
# including lemmatization (test)
test['title1_en'] = test['title1_en'].apply(lemmatize)
test['title2_en'] = test['title2_en'].apply(lemmatize)

train.dropna(inplace=True)
test.dropna(inplace=True)

# save preprocessed data 
train.to_csv('cleaned_train.csv')
test.to_csv('cleaned_test.csv')