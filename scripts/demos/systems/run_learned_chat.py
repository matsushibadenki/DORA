# ファイルパス: scripts/demos/systems/run_learned_chat.py
# 日本語タイトル: Learned SNN Chat Demo (Stable RNN Version)

import sys
import os
import torch
import torch.nn as nn
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
logging.basicConfig(level=logging.WARNING)

# --- モデル定義 (学習時と同じもの) ---
class SpikingRNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits

class SimpleTokenizer:
    def __init__(self):
        self.chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?-()[]:<>\n\"'"
        )
        self.tokens = sorted(list(set(list(self.chars)))) + ["<PAD>", "<EOS>", "<UNK>"]
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.inv_vocab = {i: t for i, t in enumerate(self.tokens)}
        self.pad_token_id = self.vocab["<PAD>"]
        self.eos_token_id = self.vocab["<EOS>"]
        self.unk_token_id = self.vocab["<UNK>"]
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        ids = []
        for c in text:
            ids.append(self.vocab.get(c, self.unk_token_id))
        ids.append(self.eos_token_id)
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        chars = []
        for i in ids:
            item = i.item()
            if item == self.eos_token_id: break
            if item == self.pad_token_id: continue
            if item == self.unk_token_id: continue
            chars.append(self.inv_vocab.get(item, ""))
        return "".join(chars)

class ConversationalSNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = SimpleTokenizer()
        self.core = SpikingRNN(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            num_layers=2
        ).to(device)
        
    def forward(self, x): return self.core(x)
    
    def generate(self, user_input: str) -> str:
        self.eval()
        # プロンプト形式を学習時と完全に一致させる
        prompt = f"Q: {user_input}\nAnswer:"
        
        input_ids = self.tokenizer.encode(prompt, max_len=64)
        valid_indices = (input_ids != self.tokenizer.pad_token_id)
        curr_ids = input_ids[valid_indices].unsqueeze(0).to(self.device)
        
        generated = []
        with torch.no_grad():
            for _ in range(100):
                logits = self.core(curr_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                generated.append(next_token.item())
        
        return self.tokenizer.decode(torch.tensor(generated))

def run_chat():
    print("=========================================")
    print(" 🧠 DORA SNN Chat (Learned Model)")
    print("=========================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = "workspace/models/chat_snn.pth"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Run training first.")
        return

    print("Loading Brain...")
    model = ConversationalSNN(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Model mismatch: {e}")
        return
        
    print("✅ Ready!")
    print("----------------------------------------")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit"]: break

            response = model.generate(user_input)
            print(f"DORA: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_chat()