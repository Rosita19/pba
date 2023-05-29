import warnings
warnings.filterwarnings('ignore')

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import yfinance as yf
# 
# st.title("Stocks App")
# symbol = st.text_input("Enter a stock symbol", "AAPL")
# if st.button("Get Quote"):
#     st.json(yf.Ticker(symbol).info)

npm install localtunnel

streamlit run app.py &>/content/logs.txt &

"""#Eksplorasi Data"""


import nltk
nltk.download('stopwords')

import pandas as pd
import numpy as np
import re
import string

data = pd.read_csv('https://raw.githubusercontent.com/Rosita19/pba/main/Tweet%20Bapak%20Jokowi%20-%20Tweet%20Bapak%20Jokowi.csv')

data.shape

data.head()

data.columns

data['Label'].value_counts()

def fix_label(before, after):
    data.loc[data['Label'] == before] = after

fix_label('negartif', 'negatif')
fix_label('netr', 'netral')
fix_label('Negatif', 'negatif')
fix_label('positi', 'positif')

data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data.Label = le.fit_transform(data["Label"])

data

"""#Preprocessing"""

#Menghapus URL agra mudah dalam melakukan pre processing data
def delete_url(text) :
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)
data['delete_url'] = data['text'].apply(lambda text: delete_url(text))
data

#Menghapus tanda baca yang ada pada dataset
tanda_baca = string.punctuation
def remove_punctuation(text) :
  return text.translate(str.maketrans('', '', tanda_baca))

data['remove_punctuation'] = data['delete_url'].apply(lambda text: remove_punctuation(text))
data

#Menghapus emoji untuk memudahkan dalam melakukan pemrosesan data
def delete_emoji(string):
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"
                          u"\U0001F300-\U0001F5FF"
                          u"\U0001F680-\U0001F6FF"
                          u"\U0001F1E0-\U0001F1FF"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags = re.UNICODE)
  return emoji_pattern.sub(r'', string)

data['delete_emoji'] = data['remove_punctuation'].apply(lambda text: delete_emoji(text))
data

#Menghapus emoticon untuk memudahkan pemrosesan data
EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}

def delete_emoticon(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

data['delete_emoticon'] = data['delete_emoji'].apply(lambda text: delete_emoticon(text))
data

#Menghapus karakter spesial, Angka, Spasi, Spasi Ganda, satu karakter
def delete_special_character(text) :
  text = text.replace('\\t', " ").replace('\\n', " ").replace('\\', "")
  text = text.encode('ascii', 'replace').decode('ascii')
  return text

def delete_number(text) :
  return re.sub(r"\d+", "", text)

# def delete_space(text) :
#   return text.strip()

# def delete_double_space(text) :
#   return re.sub('\s+', "", text)

def delete_one_character(text) :
  return re.sub(r"\b[a-zA-Z]\b", "", text)

data['delete_special_character'] = data['delete_emoticon'].apply(lambda text: delete_special_character(text))
data['delete_special_character'] = data['delete_special_character'].apply(lambda text: delete_number(text))
# data['delete_special_character'] = data['delete_special_character'].apply(lambda text: delete_space(text))
# data['delete_special_character'] = data['delete_special_character'].apply(lambda text: delete_double_space(text))
data['delete_special_character'] = data['delete_special_character'].apply(lambda text: delete_one_character(text))

data

"""#Transform Case"""

def transform_case(text):
  lowercase = text.lower()
  return lowercase

data['transform_case'] = data['delete_special_character'].apply(lambda text: transform_case(text))
data

# def clean_lower(lwr):
#     lwr = lwr.lower() # lowercase text
#     return lwr
# # Buat kolom tambahan untuk data description yang telah dicasefolding  
# data['transform_case'] = data['delete_special_character'].apply(clean_lower)
# # casefolding=pd.DataFrame(data['transform_case'])
# # casefolding
# data['transform_case']

"""#Stopword Removal"""

data['transform_case'][2]

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stopword_example = data['transform_case']
 
listStopword =  set(stopwords.words('indonesian'))
 
def stopwords_removal(text):
  return ",".join([word for word in str(text).split()if word not in listStopword])

data['stopwords_removal'] =  data['transform_case'].apply(lambda text: stopwords_removal(text))
data

"""#Stemming"""

data['transform_case'][2]

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

Fact = StemmerFactory()
Stemmer = Fact.create_stemmer()

def stemming(text):
  return " ".join([Stemmer.stem(kata) for kata in text.split(',')])

data['stemming'] =  data['stopwords_removal'].apply(lambda text: stemming(text))
data

"""# Tokenisasi"""

def tokenisasi(text):
  return text.split()

data['tokenisasi'] =  data['stemming'].apply(lambda text: tokenisasi(text))
data['tokenisasi']

join=[]
for i in range(len(data['stemming'])):
  joinkata = ''.join(data['stemming'][i])
  join.append(joinkata)

data['tokenisasi'] = pd.DataFrame(join, columns=['stemming'])
data['tokenisasi']

"""# Data Final"""

data_final = data.loc[:, ['tokenisasi', 'Label']]
data['tokenisasi']

data['Label'].value_counts()

data = data.fillna('positif')
data

"""#TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_model  = TfidfVectorizer()
tf_idf_vector = tf_idf_model.fit_transform(data['tokenisasi'])

tf_idf_array = tf_idf_vector.toarray()
fitur = tf_idf_model.get_feature_names_out()
res_tfidf = pd.DataFrame(tf_idf_array, columns=fitur)
res_tfidf

X = res_tfidf
X

"""**Save TF-IDF Result**"""

import pickle

#save TF-IDF
pickle.dump(tf_idf_model.vocabulary_,open("tfidf_result.sav","wb"))

tf_idf_model.vocabulary_

#count of feature
print(len(tf_idf_model.get_feature_names_out()))

y= data[['Label']]
y



data

"""#Splitting Data"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#gnb = GaussianNB(priors=[0.5, 0.3, 0.2], var_smoothing=1e-09)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

#y_pred = gnb.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print("Akurasi:",accuracy)
# print("Akurasi:", (round(accuracy, 2)*100),'%')

#make model prediction

input_data = ("rt megatop conannkri ccicpolri mohmahfudmd")
#lowercase_list = [s.lower() for s in input_data]
input_data = tokenisasi(input_data)

#load data
tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("tfidf_result.sav","rb"))))

#result

result = gnb.predict(loaded_vec.fit_transform([input_data]))

#seleksi kelas

if (result== 2):
  r = "positif"
elif (result == 1):
  r = "negatif"
else :
  r = "netral"

print(r)

"""#Evaluation"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

predict = gnb.predict(X_test)

cm = confusion_matrix(y_test, predict)

print(classification_report(y_test, predict))

#save model

pickle.dump(gnb,open("fixed_model.sav","wb"))
