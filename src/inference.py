from .model_loader import load_sentiment_model
from .components.preprocessing import load_vocab, Preprocessor,TokenSequenceEncoder
from config import load_config,PROJECT_ROOT,DEVICE
import torch

cfg = load_config()
vocab_path = f"{PROJECT_ROOT}/{cfg['paths']['vocab']}"
id_embedder = None
sent_model = None

def load_model_once():
    global id_embedder, sent_model

    if sent_model is None:
        vocab = load_vocab(vocab_path)
        sent_model = load_sentiment_model(vocab)
        id_embedder = TokenSequenceEncoder(cfg["seq_len"], vocab)


def inference(text:str):
    tokenized_Ids = Preprocessor().preprocess_text(text)
    if id_embedder is not None and sent_model is not None:
        embedd = id_embedder.fit_on_text(tokenized_Ids)
        embedd = torch.tensor([embedd],device=DEVICE) #should be list of Id of Tokenized words
        with torch.no_grad():
            logits = sent_model(embedd)
            pred_idx = int(torch.argmax(logits, dim=-1).cpu().numpy().item())
            sentiment = None
            if pred_idx == 0:
                sentiment = "negative"
            elif pred_idx == 1:
                sentiment = "neutral"
            else:
                sentiment = "positive"
            return pred_idx, sentiment
    else:
        raise "Model is not loaded"
        