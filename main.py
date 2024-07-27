import pandas as pd
import string
import nltk

df = pd.read_csv('SMSSpamCollection.txt',sep='\t',header=None,names=['labels','sms'])
print(df.head(5))

nltk.download('stopwords')
nltk.download('punkt')

stopwards = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation
# print(stopwards[:5])
# print(punctuation)

def pre_process(sms):
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    remove_stopwords = [word for word in tokenize if word not in stopwards]
    return remove_stopwords

df['processed'] =  df['sms'].apply(lambda x:pre_process(x))
print(df['processed'].head(5))

def catag_words():
    spam_words= []
    ham_words = []

    for sms in df['processed'][df['labels'] == 'spam']:
        for word in sms:
            spam_words.append(word)
    for sms in df['processed'][df['labels'] == 'ham']:
        for word in sms:
            ham_words.append(word)

    return spam_words,ham_words

spam_words,ham_words=catag_words()
print(spam_words[:5])
print(ham_words[:5])

def predict(sms):
    spam_counter = 0
    ham_counter = 0
    for word in sms:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    if ham_counter>spam_counter:
        accuracy = round(ham_counter/(ham_counter+spam_counter)*100)
        print(f'message is not spam with {accuracy}% ')
    elif ham_counter == spam_counter:
        print('message could be spam')
    else:
        accuracy = round(ham_counter / (ham_counter + spam_counter) * 100)
        print(f'message is  spam with {accuracy}% ')

while True:
    user_input = input('Enter Your Message: ')
    processed_input = pre_process(user_input)
    predict(processed_input)