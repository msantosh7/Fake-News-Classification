# fake-news-classification

Introduction: 

Social media has become one of the major resources for people to obtain news and information. However, large volumes of fake news or misinformation are produced online for a variety of purposes. The extensive spread of fake news/misinformation can have a serious negative impact on individuals and society. Therefore, it is important to detect fake news and misinformation in social media. 

What this project does: 

Given the title of a fake news article A and the title of a coming news article B, this model classifies B into one of the three categories:

1. agreed: B talks about the same fake news as A.
2. disagreed: B refutes the fake news in A.
3. unrelated: B is unrelated to A.

Model used: We used Linear Support Vector Classification with an oversampling SMOTE technique to account for the unbalanced labels yielded the best results, with a 74% accuracy rate and weighted F1 score.

Contents: 

1. preprocess.py - Source code for preprocessing
2. compute_similarities.py - Source code for TFIDF and Cosine similarity
3. model.py - Source code for model training and prediction
4. models_evaluation.ipynb - Source code for Evaluating different models and outputs for the same
