# SMS-Spam-Classifier


# Import Package

import common packages:

import **numpy as np**

import **pandas as pd**

import **matplotlib.pyplot as plt**

from **sklearn.model_selection** import **train_test_split**

from **sklearn.pipeline** import **Pipeline**

from **sklearn.compose** import **ColumnTransformer**

from **jcopml.utils** import **save_model, load_model**

from **jcopml.pipeline** import **num_pipe, cat_pipe**

from **jcopml.plot** import **plot_missing_value**

from **jcopml.feature_importance** import **mean_score_decrease**

from **luwiji.text_proc** import **illustration**

import Algorithm's Package:

from **nltk.tokenize** import **word_tokenize**

from **nltk.corpus** import **stopwords**

from **string** import **punctuation**

**_sw_indo = stopwords.words("indonesian") + list(punctuation)_**

from **sklearn.linear_model** import **LogisticRegression**

import Algorithm's Package:

from **sklearn.model_selection** import **RandomizedSearchCV**

from **jcopml.tuning** import **random_search_params as rsp**

from sklearn.feature_extraction.text import TfidfVectorizer

# Import Data

the data I use is a dataset belonging to a final year student named _Rahmi_ and a professor named _Wibisono_ in 2016. The title is "SMS Spam Filtering application on Android using Naive Bayes. Tut the data has been edited for project purposes. which data spam, fraud will be used as one label (label = 1). And the data contains a regular SMS labeled 0. The data consists of 1143 SMS texts, and 2 labels, namely 0 = regular sms, and 1 = spam sms.

# Explanation
# Dataset Splitting

# Training
In this case it is the same as Image Recognition, in the pipeline there is no preprocessing, because there is no need for scaling, there are no missing values ​​etc. But we need _**TFIDF**_ because it is used to encode text. So in _'prep'_ **_TFIDF_** will be entered as an encoder. We also won't use the built-in tokenizer from _scikit-learn_, because the tokenizer is just as simple as looking at spaces, periods, commas so it doesn't fit the language. And we will use the default from _NLTK_, namely _word_tokenize_.
