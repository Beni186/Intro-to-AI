from nltk.corpus import stopwords, wordnet
from nltk.tokenize.toktok import ToktokTokenizer
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)


    # Begin your code (Part 0)

    # remove 
    r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    cleanr = re.compile('<.*?>')
    preprocessed_text = re.sub(cleanr, ' ', preprocessed_text)  
    preprocessed_text = re.sub(r4,'', preprocessed_text)
    
    #lower & remove space
    preprocessed_text = preprocessed_text.strip()
    preprocessed_text = " ".join(preprocessed_text.split())
    preprocessed_text = preprocessed_text.lower()

    # lemmatization
    wnl = WordNetLemmatizer() 
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
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(preprocessed_text)
    lemmas_sent = []
    tagged_sent = pos_tag(tokens)
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    preprocessed_text = ' '.join(lemmas_sent)

    # End your code

    return preprocessed_text