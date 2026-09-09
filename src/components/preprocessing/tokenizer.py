class TokenSequenceEncoder:
    """
    The Input Dataframe needs "tokens" column to work on.
    - fit(): words to ids -> Returns dataframe[`input_ids`]
    """
    
    def __init__(self, seq_len,word_id_dict ):
        self.total_words = seq_len
        self.word_id_dict = word_id_dict
        
    def fit_df(self, dataframe):
        dataframe['input_ids'] = dataframe['tokens'].apply(self._tokens_to_ids)
        return dataframe
    
    def fit_on_text(self,text:list):
        return self._tokens_to_ids(text)
    
    #applies on Row -> Row token to id
    def _tokens_to_ids(self, tokens):
        ids = self._encode(tokens)[:self.total_words-2]
        ids = [self.word_id_dict['[CLS]']] + ids + [self.word_id_dict['[SEP]']]
        pad_len = self.total_words - len(ids)
        ids = ids + [self.word_id_dict['[PAD]']] * pad_len
        return ids
        
    def _encode(self, tokens):
        #Encode the words to id from Vocab and UNK if word isn't in Vocab
        return [self.word_id_dict.get(t, self.word_id_dict['[UNK]']) for t in tokens]
        

