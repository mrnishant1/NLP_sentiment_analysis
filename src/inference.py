from model_loader import load_sentiment_model
from components.preprocessing import load_vocab, Preprocessor,TokenSequenceEncoder
from config import load_config,PROJECT_ROOT,DEVICE
import torch

cfg = load_config()
vocab_path = f"{PROJECT_ROOT}/{cfg['paths']['vocab']}"


def inference(text:str):
    vocab = load_vocab(vocab_path)
    sentiment_model = load_sentiment_model(vocab)
    
    tokenized_Ids = Preprocessor().preprocess_text(text)
    tokenized_Ids = TokenSequenceEncoder(cfg['seq_len'],vocab).fit_on_text(tokenized_Ids)
    
    tokenized_Ids = torch.tensor([tokenized_Ids],device=DEVICE) #should be list of Id of Tokenized words
    with torch.no_grad():
        logits = sentiment_model(tokenized_Ids)
        pred_idx = int(torch.argmax(logits, dim=-1).cpu().numpy().item())
        if pred_idx == 0:
            print(logits, "negative")
        elif pred_idx == 1:
            print(logits, "neutral")
        else:
            print(logits, "positive")
        return pred_idx



if __name__ == "__main__":
    sentences = [
        "I am very happy with the excellent service.",
        "The product arrived damaged and unusable.",
        "The experience was neither good nor bad.",
        "This restaurant serves delicious food.",
        "The delivery was late and disappointing.",
        "I have mixed feelings about the new update.",
        "The movie was exciting from beginning to end.",
        "The customer support was rude and unhelpful.",
        "The results were acceptable overall.",
        "I strongly recommend this application.",
        "The battery stopped working after one day.",
        "The changes do not make much difference to me.",
        "This book was inspiring and well written.",
        "I am unhappy with the quality of the product.",
        "The weather is pleasant today.",
        "The instructions were confusing and incomplete.",
        "The concert was absolutely wonderful.",
        "The service was average for the price.",
        "I appreciate the quick response from the team.",
        "The application crashes whenever I try to open it.",
    ]

    for sentence in sentences:
        print(sentence)
        inference(sentence)