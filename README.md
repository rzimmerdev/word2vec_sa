# Word2Vec for Sentiment Analysis

This repository implements a data preprocessing script for reading information from large
text datasets, as well as implementing a Word2Vec model from nltk as well as an implementation
made in PyTorch using a Skip-Gram net.

## Preparing

Use the package manager [pip](https://pip.pypa.io/en/stable/) or Anaconda3 to install
the required packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Usage

To use the preprocessing methods import them from preprocessing.py
```python
from preprocessing import get_dataset, stem_words

df = get_dataset(name="amazon_reviews_multi", 
                 lang="en", split="train")

stem_words(df, "review_body", "english")
```

## Contributing
This repository is currently closed for contributions, except for current members, 
but feel free to use and redistribute all code for any purposes. 

## License
[MIT](https://choosealicense.com/licenses/mit/)