from datasets import load_dataset
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def get_dataset(name, lang, split):
    dataset = load_dataset(name, lang)
    print(dataset[split].shape)
    print(type(dataset[split].features))
    dataset_train = pd.DataFrame.from_dict(dataset[split].shuffle()[:10000])

    return dataset_train


def preprocess(dataframe, col, lang):
    dataframe['tokenized'] = dataframe[col].map(word_tokenize)

    stemmer = SnowballStemmer(lang)

    dataframe['stemmed'] = dataframe['tokenized'].map(lambda text: [stemmer.stem(word) for word in text])


if __name__ == "__main__":
    df = get_dataset(name="amazon_reviews_multi", lang="en", split="train")
    preprocess(df, "review_body", "english")

    print(df[['review_body', 'stars', 'stemmed']])
