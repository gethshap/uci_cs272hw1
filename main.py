from speech import *

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2,f_regression

pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf",LogisticRegression(max_iter=1000)),
    ]
)

parameters = {
    "vect__max_df": [0.17],
    'vect__max_features': [1500],
    "vect__ngram_range": [(1, 1)],
    'tfidf__use_idf': (True, False),
    #'tfidf__use_idf': [True],
    #'tfidf__norm': ('l1', 'l2')
    'tfidf__norm':[ 'l2'],
}

def tfidf(data):
    vectorizer = CountVectorizer() # This class will transform the words in the text into a word frequency matrix, and the matrix element a[i][j] represents the word frequency of word j under class i text
    transformer = TfidfTransformer() # This class will count the tf-idf weights of each word
    tfidf_before = vectorizer.fit_transform(data)
    tfidf = transformer.fit_transform(tfidf_before)
    return tfidf


if __name__ == "__main__":
    import tarfile

    tar = tarfile.open("data/speech.tar.gz", "r:gz")


    class Data: pass


    speech = Data()
    print("-- train data")
    speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
    print(len(speech.train_data))
    print("-- dev data")
    speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
    print(len(speech.dev_data))
    print("-- transforming data and labels")


    grid_search = GridSearchCV(pipeline, parameters, verbose=3,error_score='raise')
    grid_search.fit(speech.train_data, speech.train_labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    b =  grid_search.cv_results_['mean_test_score']
    #write_pred_kaggle_file(unlabeled,grid_search,"data/myspeech-pred.csv",speech)

    print(str(b))
    '''
    print("Reading unlabeled data")
    unlabeled = read_unlabeled("data/speech.tar.gz", speech)
    transformer = SelectKBest(k=1000,score_func=chi2)
    transformed_train_data = transformer.fit_transform(speech.train_data, speech.train_labels)
    transformed_unlabeled = transformer.transform(unlabeled.X)
    print('f')
    
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline
    x = [ 100, 200,300, 1000, 2000, 3000, 4000,5000]
    y = b
    model=make_interp_spline(x, y[1:])
    xs = np.linspace(0, 5000, 5000)
    ys = model(xs)
    plt.plot(xs, ys)
    plt.xlabel("#feature")
    plt.ylabel("accuracy")
    plt.show()



