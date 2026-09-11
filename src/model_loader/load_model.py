import torch
from config import load_config,DEVICE,PROJECT_ROOT
cfg = load_config()
from ..components.model import MiniBERT,SentimentModel
 
NUM_CLASSES = cfg["sentiment"]["num_classes"]
vocab_path = f"{PROJECT_ROOT}/{cfg['paths']['vocab']}"

sentiment_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['sentiment_checkpoint']}"
mlm_checkpoint = f"{PROJECT_ROOT}/{cfg['paths']['mlm_checkpoint']}"

def load_sentiment_model(vocab)->SentimentModel:
    """
    Returns-> Sentiment_model in .eval() mode. 
    """
    
    model = MiniBERT(
        vocab_size=len(vocab),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        seq_len=cfg["seq_len"],
        ff_hidden=cfg["ff_hidden"],
        pad_idx=vocab["[PAD]"],
    ).to(DEVICE)

    backbone_state = torch.load(mlm_checkpoint,map_location=DEVICE,weights_only=True)
    model.load_state_dict(backbone_state)

    sentiment_model = SentimentModel(model,num_classes=NUM_CLASSES,).to(DEVICE)
    classifier_state = torch.load(sentiment_checkpoint,map_location=DEVICE,weights_only=True)
    sentiment_model.classifier.load_state_dict(classifier_state)

    sentiment_model.eval()
    return sentiment_model