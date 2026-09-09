from pathlib import Path
import numpy as np
import pandas as pd
from components.preprocessing import Preprocessor,TokenSequenceEncoder
from config import load_config,DEVICE,PROJECT_ROOT
cfg = load_config()
from components.eval.evaluate import evaluate
from components.preprocessing.vocab import load_vocab
NUM_CLASSES = cfg["sentiment"]["num_classes"]
vocab_path = f"{PROJECT_ROOT}/{cfg['paths']['vocab']}"
sentiment_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['sentiment_checkpoint']}"
mlm_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['mlm_checkpoint']}"
from model_loader import load_sentiment_model

#================================Evaluate====================
def test_pipeline():
    """
    Returns {accuracy, confusion matrix}
    """    
    vocab = load_vocab(vocab_path)
    sentiment_model = load_sentiment_model(vocab)

    df = pd.read_csv(f"{PROJECT_ROOT}/{cfg['paths']['test_data']}")
    df = Preprocessor().initiate_dataprocessing(df)
    
    if Path(vocab_path).is_file():
        tokenizer = TokenSequenceEncoder(cfg['seq_len'],vocab)
        df = tokenizer.fit_df(df)
    else:
        raise FileNotFoundError(f"Missing tokenizer vocabulary: {vocab_path}")
    
    X_test = np.stack(df['input_ids'].values)
    Y_test = df['sentiment_label']  
    return evaluate(X_test,Y_test,DEVICE,sentiment_model)
 
    
if __name__ == "__main__":
    test_pipeline()
    