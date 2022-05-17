# SMS-Spam-Classifier
# Natural Language Processing
### what is natural language processing?
Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. Together, these technologies enable computers to process human language in the form of text or voice data and to ‘understand’ its full meaning, complete with the speaker or writer’s intent and sentiment.

NLP drives computer programs that translate text from one language to another, respond to spoken commands, and summarize large volumes of text rapidly—even in real time. There’s a good chance you’ve interacted with NLP in the form of voice-operated GPS systems, digital assistants, speech-to-text dictation software, customer service chatbots, and other consumer conveniences. But NLP also plays a growing role in enterprise solutions that help streamline business operations, increase employee productivity, and simplify mission-critical business processes.

### Basic Text Preprocessing
#### Normalization
change all letters to uppercase or lowercase.
#### Tokenization
tokenization will break the entire text into words, or sentences.
#### Punctuation Removal
punctuation eraser
#### Alphanumeric Cleansing
just make sure that the variable consists of letters and numbers.
#### Stopwords Removal
are words that often / always appear in conversational language, stopwords is not absolute and depends on what the domain is talking about.
### Why Preprocessing
Vocabuary will be used as a feature. Example(in indonesian language):

- Ini adalah pensil
- Ini adalah pulpen
- Saya beli pensin ini    
- Saya beli pulpen itu
 
Tokens:
- Ini 
- adalah
- pensil
- pulpen
- saya
- beli
- itu
This means that only with these tokens we can make the text sentence above.
Then the token/vovabuary will be compiled into a table.

![image](https://user-images.githubusercontent.com/86812576/168838238-fb84353c-5747-4821-9e41-4a83ded2920a.png)

The left side is as a line (document). And the right part is the feature(vocab) taken from the token.
If there is the word "Hello" with "hello" is a different token even though the meaning is the same, unless it is normalized.

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

from **sklearn.model_selection** import **RandomizedSearchCV**

from **jcopml.tuning** import **random_search_params as rsp**

from **sklearn.feature_extraction.text** import **TfidfVectorizer**

# Import Data

the data I use is a dataset belonging to a final year student named _Rahmi_ and a professor named _Wibisono_ in 2016. The title is "SMS Spam Filtering application on Android using Naive Bayes. Tut the data has been edited for project purposes. which data spam, fraud will be used as one label (label = 1). And the data contains a regular SMS labeled 0. The data consists of 1143 SMS texts, and 2 labels, namely 0 = regular sms, and 1 = spam sms.

# Explanation
# Dataset Splitting

Because the data only has 2 columns then just do it, X = Text, and y = label.

![xy](https://user-images.githubusercontent.com/86812576/168048199-4497d05a-a480-4a6a-9be9-5250f84ddfcc.png)

And then split data into train, and test.

# Training
In this case it is the same as Image Recognition, in the pipeline there is no preprocessing, because there is no need for scaling, there are no missing values ​​etc. But we need _**TFIDF**_ because it is used to encode text. So in _'prep'_ **_TFIDF_** will be entered as an encoder. I also won't use the built-in tokenizer from _scikit-learn_, because the tokenizer is just as simple as looking at spaces, periods, commas so it doesn't fit the language. And we will use the default from _NLTK_, namely _word_tokenize_. For stop_words I use sw_indo, ideally we should make our own but we will use the existing one in Indonesian language.

I use Logistic Regression and Random Search Algorithm with cross validation = 3, and n_iter = 50. Then we just run it.

![Screenshot 2022-05-12 173906](https://user-images.githubusercontent.com/86812576/168052660-0c49c71d-ccdb-48b7-8578-01ba6062436b.png)

In this case the model is quite good. Next I will do a sanity check to make sure it's really good. in the end the test data is just a simulation, so we split it for future simulations if we have original data in our mobile it's good to see if our model overfit or not.

# Sanity Check

In sanity check I entered an sms which is actually spam, this sms used to be received by most Indonesians. I don't know where it is came from, it's like a random sms sent to many people. we will try this sms to our model whether our model can predict correctly.

This is the content of the sms.

![Screenshot 2022-05-12 175120](https://user-images.githubusercontent.com/86812576/168054795-e355dce6-f16c-4091-aee0-79973820f290.png)

And the results are indeed good, our model successfully predicts that the sms is spam. but we will try some other sms whether our model is good.

I will enter the second sms, this is the original sms (not spam) we will see if the model can predict correctly.

![Screenshot 2022-05-12 175520](https://user-images.githubusercontent.com/86812576/168055574-f8d9485c-9856-4358-9621-d477d4ab43ea.png)

The result is our model can predict correctly the second sms. I will try to enter the last sms which is a spam sms.

![Screenshot 2022-05-12 175956](https://user-images.githubusercontent.com/86812576/168056047-849086c8-9c0f-4508-b52f-e9070a3475b3.png)

And it turns out our model is really good. because it managed to predict 3 sms tested whether spam or not.

### lbfgs

