from datasets import load_dataset
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import download


def get_dataset(name, lang, split):
    dataset = load_dataset(name, lang)
    print(dataset[split].shape)
    print(type(dataset[split].features))
    dataset_train = pd.DataFrame.from_dict(dataset[split].shuffle()[:10000])

    return dataset_train


def stem_words(dataframe, col, lang):
    dataframe['tokenized'] = dataframe[col].map(word_tokenize)

    stemmer = SnowballStemmer(lang)

    dataframe['stemmed'] = dataframe['tokenized'].map(lambda text: [stemmer.stem(word) for word in text])


def maximum_abs_scale(dataframe, column):
    dataframe[column] = dataframe[column] / dataframe[column].abs().max()


if __name__ == "__main__":
    download('punkt')
    df = get_dataset(name="amazon_reviews_multi", lang="en", split="train")
    stem_words(df, "review_body", "english")

    maximum_abs_scale(df, 'stars')

    df[['review_body', 'stars', 'stemmed']].to_csv('../sets/stemmed.csv', quotechar='"', index=False)
