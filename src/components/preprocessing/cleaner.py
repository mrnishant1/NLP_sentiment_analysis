import pandas as pd
import string
# nltk.download('stopwords')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class Preprocessor:
    """Clean and tokenize text from a DataFrame.
    The DataFrame must contain a column named ``clean_comment``.
    - Returns `tokens` column
    """

    def __init__(self):
        self.TEXT_LABEL = 'clean_comment'
        self.CLASSIFY = 'category'
        self.ps = PorterStemmer()
        self.en_stopwords = set(stopwords.words('english'))
        
    def initiate_dataprocessing(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        """Clean comments and add a tokenized ``tokens`` column, returns DataFrame copy."""
        df = dataframe.copy()
        if self.TEXT_LABEL not in dataframe.columns or self.CLASSIFY not in dataframe.columns:
            raise ValueError("Missing required column: clean_comment or category")
        df = df.dropna(subset=[self.TEXT_LABEL]).reset_index(drop=True)
        df['clean_comment'] = df['clean_comment'].str.lower() #lower case
        df['clean_comment'] = df['clean_comment'].apply(self._remove_punc) #remove punctuation symbols
        df['tokens'] = df['clean_comment'].apply(self._preprocess_tokens) #nltk word tokenizer
        df['sentiment_label'] = df[self.CLASSIFY].astype(int) + 1
        return df
    
    def preprocess_text(self, text:str)->list:
        text = text.lower()
        text = self._remove_punc(text)
        text = self._preprocess_tokens(text)
        return text
    
    def _preprocess_tokens(self,text):
        tokens = word_tokenize(text) #nltk word tokenizer from sentences -> words
        tokens = [self.ps.stem(t) for t in tokens if t not in self.en_stopwords]
        return tokens
        
    def _remove_punc(self,text:str)-> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
