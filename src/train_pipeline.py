import torch
from pathlib import Path
import numpy as np
import pandas as pd
from components import DataIngestion
from components.preprocessing import Preprocessor,TokenSequenceEncoder
from config import load_config,DEVICE,PROJECT_ROOT
cfg = load_config()
from components.model import MiniBERT,train_mlm,SentimentModel,train_sentiment
from components.eval.evaluate import evaluate
import torch.nn as nn 
from components.preprocessing.vocab import create_vocab,save_vocab,load_vocab
NUM_CLASSES = cfg["sentiment"]["num_classes"]
vocab_path = f"{PROJECT_ROOT}/{cfg['paths']['vocab']}"
sentiment_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['sentiment_checkpoint']}"
mlm_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['mlm_checkpoint']}"

def train_pipeline():
    """
    Steps:
        - Load Validated_data,train,test,valid paths
        - Preprocess, tokenize train_data
        - token_to_id
        - Create Model Object
            - Train or Load weights
        - Sentiment Model Object
            - Train or Load Weights
        - Preprocess, tokenize test_data-> cls representation-> sentiment prediction -> Score
    """
    
    #Preprocessing ==================================================
    train_path = DataIngestion().initiate_data_ingestion()["train_dataset"]
    df = pd.read_csv(train_path) 
    df = Preprocessor().initiate_dataprocessing(df)
    
    
    #Old= Prev training vocabulary || New = New data + old vocab
    old_word_id_dict = load_vocab(vocab_path) if Path(vocab_path).is_file() else None
    new_word_id_dict = create_vocab(df,old_word_id_dict)
    save_vocab(vocab_path,new_word_id_dict)
    
    #Tokens -> Id
    tokenizer = TokenSequenceEncoder(cfg["seq_len"],new_word_id_dict)
    df = tokenizer.fit_df(df)
    
    
    #Train Input/Output ==================================================
            
    X_train = np.stack(df['input_ids'].values)
    Y_train = np.stack(df['sentiment_label'])    
    class_counts = np.bincount(Y_train, minlength=NUM_CLASSES)
    # Class-weighted loss so the majority/minority classes are treated fairly
    class_weights = torch.tensor(class_counts.sum() / (NUM_CLASSES * class_counts), dtype=torch.float32, device=DEVICE)
    
    print("Class weights:", class_weights)
    

    #===========================================Model Init=========================================
    word_id_dict = tokenizer.word_id_dict
    
    mlm_state_dict = torch.load(mlm_checkpoint) if Path(mlm_checkpoint).is_file() else None
        
    model = MiniBERT(
        vocab_size=len(word_id_dict),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        seq_len=cfg["seq_len"],
        ff_hidden=cfg["ff_hidden"],
        pad_idx=word_id_dict["[PAD]"],
    ).to(DEVICE)

    sentiment_model = SentimentModel(model, num_classes=NUM_CLASSES).to(DEVICE)

    if mlm_state_dict is not None:
        try:
            model.load_state_dict(mlm_state_dict)
        except RuntimeError:
            model = load_n_resize_model_(
                model,
                mlm_state_dict,
                old_word_id_dict or {},
                new_word_id_dict,
                cfg["d_model"],
            )

    # Load previous sentiment-classifier weights if available
    sent_state_dict = torch.load(sentiment_checkpoint) if Path(sentiment_checkpoint).is_file() else None
    
    if sent_state_dict is not None:
        try:
            sentiment_model.classifier.load_state_dict(sent_state_dict)
        except RuntimeError:
            # Safe fallback: keep the current classifier if architecture changes.
            pass
    
    #========================== TRAIN ======================================
    mlm_loss_history = train_mlm(model, X_train, cfg["mlm"]["epochs"], cfg["mlm"]["batch_size"], cfg["mlm"]["lr"])
            
    sentiment_loss_history = train_sentiment(
        sentiment_model, X_train, Y_train,
        cfg["sentiment"]["epochs"], cfg["sentiment"]["batch_size"], cfg["sentiment"]["lr"], class_weights
    )
    
    torch.save(model.state_dict(), mlm_checkpoint)
    torch.save(
        sentiment_model.classifier.state_dict(),
        sentiment_checkpoint,
    )
    
    #===========================Evaluating====================================
    Y_test = df['sentiment_label']
    evaluate(X_train,Y_test,DEVICE,sentiment_model)

    return mlm_loss_history,sentiment_loss_history

def load_n_resize_model_(model:MiniBERT,prev_loadstate_dict, old_word_to_id, new_word_to_id, d_model):
    """
    new_word_to_id: old_vocab + new_vocab
    model: miniBert model
    d_model: features
    """
    
    #embedding.weights miss-match will happen when try to train with different vocab
    #mlm_head.weight, mlm_head.bias miss-match will happen
    #it should Incorporate old embedding.weights + new embeddings
    
    old_embedding_weights = prev_loadstate_dict['embedding.weight'].data
    old_mlm_head_weights = prev_loadstate_dict['mlm_head.weight'].data
    old_mlm_head_bias = prev_loadstate_dict['mlm_head.bias'].data
    
    vocab_size = len(new_word_to_id)
    print("vocab size---------------",vocab_size)
    new_embedding_weights = torch.rand(vocab_size,d_model) *0.2
    new_mlm_head_weights = torch.rand(vocab_size,d_model) *0.2
    new_mlm_head_bias = torch.rand(vocab_size)
    
    for word, old_id in old_word_to_id.items():
        if word in old_word_to_id:
            new_id = old_word_to_id[word]
            new_mlm_head_weights[new_id] = old_mlm_head_weights[old_id]
            new_mlm_head_bias[new_id] = old_mlm_head_bias[old_id]
            new_embedding_weights[new_id] = old_embedding_weights[old_id]
    
    model.embedding = nn.Embedding(vocab_size,d_model,padding_idx=new_word_to_id['[PAD]'], _freeze = False)
    model.mlm_head = nn.Linear(d_model, vocab_size)
    
    model.embedding.weight.data = new_embedding_weights.data
    model.mlm_head.weight.data = new_embedding_weights.data
    model.mlm_head.bias.data = new_mlm_head_bias.data
    return model
    
    
if __name__ == "__main__":
    mlm_loss_history,sentiment_loss_history = train_pipeline()
    print(mlm_loss_history,sentiment_loss_history)