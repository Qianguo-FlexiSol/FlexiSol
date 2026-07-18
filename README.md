1.INSTALL
 
python -c
conda env create -f env.yml
conda activate solubility
"from transformers import AutoTokenizer, EsmModel
m='facebook/esm2_t33_650M_UR50D'
AutoTokenizer.from_pretrained(m)
EsmModel.from_pretrained(m)
"

2.download the ESM‑2 model

python -c "from transformers import AutoTokenizer, EsmModel; \
           m='facebook/esm2_t33_650M_UR50D'; \
           AutoTokenizer.from_pretrained(m); \
           EsmModel.from_pretrained(m)"

3.USAGE

The repository includes three core Python scripts:

train_nn.py – defines the MLP model and training routines.

extract_esm8m.py – loads the ESM‑2 tokenizer and model.

predict_single.py – a ready‑to‑use script for single‑sequence prediction.

Quick single‑sequence prediction
Edit the seq variable in predict_single.py (currently set to an example sequence) and run:

bash
python predict_single.py
Example output:

text
Single prediction: 0.8732 Soluble
The default threshold for solubility is 0.6 (probability > 0.6 → Soluble). You can adjust this threshold in the script.

4.Using the Python API
You can also import the prediction functions into your own code:

python
!git clone https://github.com/SeqSol-qianguo/SeqSol.git
%cd SeqSol

!pip install -q transformers torch

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

class MLP(nn.Module):
    def __init__(self, input_dim=1280, hidden1=256, hidden2=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),   # index 0
            nn.ReLU(),                       # index 1
            nn.Dropout(0.2),                 # index 2
            nn.Linear(hidden1, hidden2),     # index 3
            nn.ReLU(),                       # index 4
            nn.Dropout(0.2),                 # index 5
            nn.Linear(hidden2, output_dim)   # index 6
        )
    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
print(
    "Loading the ESM-2 model "
    "(approximately 2.4 GB will be downloaded on the first run; please be patient)..."
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
esm_model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

print("ESM-2 model loaded successfully.")

def seq2feat(sequence):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = esm_model(**inputs)

    # Return the [CLS] token embedding with shape (1, 1280)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

WEIGHT_PATH = "/content/SeqSol/best_esm650m_nn.pth"
state_dict = torch.load(WEIGHT_PATH, map_location="cpu")

in_dim = state_dict["net.0.weight"].shape[1]      # 1280
hidden1 = state_dict["net.0.weight"].shape[0]     # 256
hidden2 = state_dict["net.3.weight"].shape[0]     # 128
out_dim = state_dict["net.6.weight"].shape[0]     # 1

print(
    f"MLP dimensions: input {in_dim} → hidden layer 1 {hidden1} "
    f"→ hidden layer 2 {hidden2} → output {out_dim}"
)

mlp = MLP(input_dim=in_dim, hidden1=hidden1, hidden2=hidden2, output_dim=out_dim).to(DEVICE)
mlp.load_state_dict(state_dict)
mlp.eval()
print("MLP success")

def predict_solubility(sequence):
    feat = seq2feat(sequence)                     # (1, 1280)
    feat_tensor = torch.tensor(feat).float().to(DEVICE)
    with torch.no_grad():
        logit = mlp(feat_tensor).item()          
        prob = torch.sigmoid(torch.tensor(logit)).item()
    return prob

print("\n" + "=" * 50)
print("Protein Solubility Prediction Tool")
print(
    "Enter a protein sequence using single-letter amino acid codes "
    "(e.g., MLSRAVCGTSRQLAPVLAYLGSRQ)"
)
print("Enter an empty line or type 'quit' to exit")
print("=" * 50)

while True:
    seq = input("\nEnter sequence: ").strip()

    if not seq or seq.lower() == "quit":
        print("Exiting the program.")
        break

    # Perform a basic amino acid composition check (warning only)
    allowed = set("ACDEFGHIKLMNPQRSTVWY")

    if not set(seq.upper()).issubset(allowed):
        print(
            "Warning: The sequence contains non-standard amino acid letters. "
            "Prediction will still be attempted..."
        )

    try:
        prob = predict_solubility(seq)
        pred = "Soluble" if prob > 0.6 else "Insoluble"

        print(
            f"Result: solubility probability = {prob:.4f} "
            f"→ {pred}"
        )

    except Exception as e:
        print(f"Prediction error: {e}")
		
# Example
prob = predict_solubility("MLSRAVCGTSRQLAPVLAYLGSRQ")
print(f"Probability: {prob:.4f} → {'Soluble' if prob > 0.6 else 'Insoluble'}")

5.Citation
If you use SeqSol in your research, please cite:

bibtex
@software{SeqSol2026,
  author = {SeqSol-qianguo},
  title = {SeqSol: Protein Solubility Prediction Tool based on ESM-2},
  year = {2026},
  url = {https://github.com/SeqSol-qianguo/SeqSol}
}
