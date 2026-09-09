def create_vocab(dataframe,word_id_dict:dict=None):
    if word_id_dict is not None:
        unique_words = set()
        for toks in dataframe['tokens']:
            unique_words.update(toks)
        #We don't add special words to new vocab, neither we change old words id's 
        #Just add new words without changing old one's
        lastItemKey = next(reversed(word_id_dict.values()))+1
        for i in unique_words:
            if i not in word_id_dict:
                word_id_dict[i] = lastItemKey
            lastItemKey+=1
        return word_id_dict
        
    else:
        unique_words = set()
        for toks in dataframe['tokens']:
            unique_words.update(toks)
        vocabulary = ['[PAD]', '[CLS]', '[UNK]', '[SEP]', '[MASK]'] + sorted(unique_words)
        word_id_dict = {w: i for i, w in enumerate(vocabulary)}
        return word_id_dict

def save_vocab(path,word_id_dict):
    import json
    json.dump(word_id_dict, open(path, 'w'))

def load_vocab(path):
    import json
    with open(path, 'r') as file:
        return json.load(file)