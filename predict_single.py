# predict_single.py
import torch, numpy as np
from train_nn import MLP, DEVICE
from extract_esm8m import MODEL_NAME, AutoTokenizer, EsmModel

BEST_PT = "best_esm650m_nn.pth"

# loading tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
esm = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def seq2feat(seq):
    inp = tok(seq, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        return esm(**inp).last_hidden_state[:, 0].cpu().numpy()

def predict_seq(seq, model):
    feat = torch.tensor(seq2feat(seq)).float().to(DEVICE)
    with torch.no_grad():
        prob = model(feat).item()
    return prob

def load_model(d_in):
    m = MLP(d_in).to(DEVICE)
    m.load_state_dict(torch.load(BEST_PT, map_location=DEVICE))
    m.eval()
    return m

if __name__ == "__main__":
    seq = ""
    model = load_model(seq2feat(seq).shape[1])
    p = predict_seq(seq, model)
    print("Single prediction:", round(p, 4), "Soluble" if p > 0.6else "Insoluble")
