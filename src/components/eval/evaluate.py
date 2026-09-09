import torch
from components.model import SentimentModel, MiniBERT
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(all_input_ids, Y_test, device, sentiment_model: SentimentModel):
    sentiment_model.eval()
    
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(all_input_ids), 256):
            #Input
            x = torch.tensor(all_input_ids[start:start + 256], dtype=torch.long, device=device)
            #Output
            logits = sentiment_model(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    acc = accuracy_score(Y_test, all_preds)
    cm = confusion_matrix(Y_test, all_preds)
    print("Accuracy:", acc)
    print("Confusion Matrix (rows=true, cols=pred, order = [-1, 0, 1] -> [0,1,2]):")
    print(cm)
    return acc, cm