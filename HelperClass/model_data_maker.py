import warnings
warnings.filterwarnings("ignore")
from flask import make_response

import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from autocorrect import Speller


spell = Speller(lang = 'en')


# function to correct spellings (english)
def spell_check(sentence):
    ''' This function is used to correct the words in the text'''
    word_list = []
    for word in sentence.strip().split():
        word = spell(word)  # correct each word in a sentence
        word_list.append(word)

    return " ".join(word_list)  # return corrected sentence


# Load the sgd model from disk and use for predictions
def classify_utterance(utt):

    # load the model
    # change file location if your model store any where
    loaded_model = pickle.load(open('HelperClass/sgd_model_price.model', 'rb'))

    # make a prediction
    return loaded_model.predict(utt)


def pre_process_data(input_data):
    # convert input data to dataframe for better preprocess data
    res = {}
    try:
        input_df = pd.DataFrame([input_data])
    except Exception as err:
        print(err)
        return make_response("Something went Wrong to converting data frame", 404)
    null_exist = input_df.isnull().values.any()
    if null_exist:
        res['msg'] = "Null Value exist in input"
        return make_response(res, 404)

    X_train  =  input_df  # taken input as a train set so you can change it but i am  taking your variable names
    preprocessed_model_info = []
    for sent in tqdm(X_train['Model_Info'].values):
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = spell_check(sent)
        preprocessed_model_info.append(sent.strip())

    X_train['preprocessed_model_info'] = preprocessed_model_info

    # as per our observation we replace the phone (iphone after spell correction becomes phone) with name
    X_train['preprocessed_model_info'] = X_train['preprocessed_model_info'].str.replace('phone', 'name0')

    # apply the preprocessing steps on this feature just like the previous feature
    preprocessed_add_desc = []

    for sent in tqdm(X_train['Additional_Description'].values):
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = " ".join(
            filter(lambda x: x[:5] != '10100', sent.split()))  # removes all encoded words starting with '10100'
        sent = spell_check(sent)
        preprocessed_add_desc.append(sent.strip())

    X_train['preprocessed_add_desc'] = preprocessed_add_desc

    ## Text Feature encoding
    # one-hot encoding 'Brand'

    encoder = OneHotEncoder()
    encoder.fit(X_train['Brand'].values.reshape(-1, 1))

    X_train_brand = encoder.fit_transform(X_train['Brand'].values.reshape(-1, 1))

    # One-hot encoding 'Locality'

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train['Locality'].values.reshape(-1, 1))

    # 'City'

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train['City'].values.reshape(-1, 1))

    X_train_city = encoder.transform(X_train['City'].values.reshape(-1, 1))

    # 'State'

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train['State'].values.reshape(-1, 1))

    X_train_state = encoder.transform(X_train['State'].values.reshape(-1, 1))

    # 'preprocessed_model_info' BOW encoding
    vectorizer = CountVectorizer(min_df = 0, ngram_range = (1,4))#min_df=1, ngram_range=(1, 4)

    X_train_model_info_bow = vectorizer.fit_transform(X_train['preprocessed_model_info'].values)



    vectorizer = TfidfVectorizer(min_df=0, ngram_range=(1, 4))#min_df=3, ngram_range=(1, 4)

    # we use the fitted CountVectorizer to convert the text to vector
    X_train_model_info_tfidf = vectorizer.fit_transform(X_train['preprocessed_model_info'].values)

    # 'preprocessed_add_desc' BOW encoding

    vectorizer = CountVectorizer(min_df=0, ngram_range=(1, 4))#min_df=5, ngram_range=(1, 4)

    # we use the fitted CountVectorizer to convert the text to vector
    X_train_add_desc_bow = vectorizer.fit_transform(X_train['preprocessed_add_desc'].values)

    # bow encoded text features
    # drop locality from the list due to very high variation

    from scipy.sparse import hstack

    X_train_bow = hstack((X_train_brand, X_train_city, X_train_state,
                          X_train_model_info_bow, X_train_add_desc_bow))



    # predict price
    predicted_price = classify_utterance(X_train_bow)
    res = {}
    res['msg'] = "Price Predicted Successfully"
    res['data'] = {'Predicted Price': predicted_price[0]}
    return res
