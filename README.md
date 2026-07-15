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
print("正在加载 ESM-2 模型（首次需下载约 2.4GB，请耐心等待）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
esm_model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("ESM-2 模型加载完成。")

def seq2feat(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    # 返回 [CLS] token 的嵌入 (1, 1280)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

WEIGHT_PATH = "/content/SeqSol/best_esm650m_nn.pth"
state_dict = torch.load(WEIGHT_PATH, map_location="cpu")

in_dim = state_dict["net.0.weight"].shape[1]      # 1280
hidden1 = state_dict["net.0.weight"].shape[0]     # 256
hidden2 = state_dict["net.3.weight"].shape[0]     # 128
out_dim = state_dict["net.6.weight"].shape[0]     # 1

print(f"MLP 维度：输入 {in_dim} → 隐藏1 {hidden1} → 隐藏2 {hidden2} → 输出 {out_dim}")

mlp = MLP(input_dim=in_dim, hidden1=hidden1, hidden2=hidden2, output_dim=out_dim).to(DEVICE)
mlp.load_state_dict(state_dict)
mlp.eval()
print("MLP 权重加载成功。")

def predict_solubility(sequence):
    feat = seq2feat(sequence)                     # (1, 1280)
    feat_tensor = torch.tensor(feat).float().to(DEVICE)
    with torch.no_grad():
        logit = mlp(feat_tensor).item()           # 线性层输出（未经 sigmoid）
        prob = torch.sigmoid(torch.tensor(logit)).item()
    return prob

print("\n" + "="*50)
print("蛋白质可溶性预测工具")
print("请输入单字母表示的蛋白质序列（例如：MLSRAVCGTSRQLAPVLAYLGSRQ）")
print("输入空行或 'quit' 退出")
print("="*50)

while True:
    seq = input("\n请输入序列: ").strip()
    if not seq or seq.lower() == 'quit':
        print("退出程序。")
        break
    # 简单检验氨基酸组成（仅警告）
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(seq.upper()).issubset(allowed):
        print("警告：序列含有非标准氨基酸字母，仍尝试预测...")
    try:
        prob = predict_solubility(seq)
        pred = "可溶性 (Soluble)" if prob > 0.6 else "不可溶性 (Insoluble)"
        print(f"结果：可溶性概率 = {prob:.4f}  →  {pred}")
    except Exception as e:
        print(f"预测出错：{e}")
		
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
