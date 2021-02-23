import re
import string

import nltk

stopwords = nltk.corpus.stopwords.words('english')

# Download the Wordnet package if not available, it is required by Lemmatizer
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()


def remove_punctuation(text):
    """
    Function to remove whitespace and punctuation from the passed text
    :param text: Text to remove punctuation
    :return: Text with punctuation removed
    """
    return ' '.join([word for word in re.sub(r'\s+', ' ', text).split() if word not in string.punctuation])


def tokenize_stopwords(text):
    """
    Function the stopwords from the passed text
    :param text: Text to remove stopwords
    :return: tokens of words
    """
    tokens = [word for word in text.split() if word not in stopwords]
    return tokens


def stem_tokens(tokens):
    """
    Function to apply stemming on the passed token of words
    :param tokens:
    :return: stemming results
    """
    return [ps.stem(word) for word in tokens]


def lammatize_tokens(tokens):
    """
    # Function to perform stemming on the passed token of Words
    :param tokens:
    :return: lamentized results
    """
    return [wn.lemmatize(word) for word in tokens]


def remove_emoji(text):
    """
    Remove any emoji characters from the text
    :param text: Text to clean
    :return: Clean text
    """
    import emoji
    return re.sub(r':', '', emoji.demojize(text))


def remove_url(text):
    return re.sub(r'http\S+', '', text)  # Remove URLs


def remove_dollar(text):
    return re.sub(r'\$\S+', 'dollar', text)  # Change dollar amounts to dollar


def remove_special_chars(text):
    """
    Remove any characters except words and number from passed text.
    :param text: Text to remove characters
    :return: cleaned text.
    """
    return re.sub(r'[^\w\s]', '', text).strip()


# Function to remove punctuation and Tokenize
def clean_text(text):
    text = remove_punctuation(text)
    tokens = tokenize_stopwords(text)
    text = ' '.join(lammatize_tokens(tokens))
    return text


if __name__ == '__main__':
    text = "Hello, how are you doing. I am doing gr8 !!! @ ; "
    print(text)
    print(remove_punctuation(text))
