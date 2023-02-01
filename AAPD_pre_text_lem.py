from stop_words import stop_words
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    punc = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    string = re.sub(r"[%s]+" %punc,"",string)
    string = re.sub(r"[^A-Za-z0-9`]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"(\d+)", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
#  获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
# text_train = []
# with open('AAPD/text_train') as f:
#     for line in f.readlines():
#         lemmas_sent = []
#         temp = clean_str(line)
#         words = []
#         tokens = word_tokenize(temp)
#         tagged_sent = pos_tag(tokens)
#         for tag in tagged_sent:
#             wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
#             lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
#         for word in lemmas_sent:
#             if word not in stop_words:
#                 words.append(word)
#         doc_str = ' '.join(words).strip()
#         text_train.append(doc_str)
# clean_corpus_str_train = '\n'.join(text_train)


text_val = []
with open('AAPD/text_val') as f:
    for line in f.readlines():
        lemmas_sent = []
        temp = clean_str(line)
        words = []
        tokens = word_tokenize(temp)
        tagged_sent = pos_tag(tokens)
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        for word in lemmas_sent:
            if word not in stop_words:
                words.append(word)
        doc_str = ' '.join(words).strip()
        text_val.append(doc_str)
clean_corpus_str_val = '\n'.join(text_val)


text_test = []
with open('AAPD/text_test') as f:
    for line in f.readlines():
        lemmas_sent = []
        temp = clean_str(line)
        words = []
        tokens = word_tokenize(temp)
        tagged_sent = pos_tag(tokens)
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        for word in lemmas_sent:
            if word not in stop_words:
                words.append(word)
        doc_str = ' '.join(words).strip()
        text_test.append(doc_str)
clean_corpus_str_test = '\n'.join(text_test)




# with open('AAPD/clean_text_train_lem.txt', 'w') as f:
#     f.write(clean_corpus_str_train)

with open('AAPD/clean_text_val_lem.txt', 'w') as f:
    f.write(clean_corpus_str_val)

with open('AAPD/clean_text_test_lem.txt', 'w') as f:
    f.write(clean_corpus_str_test)