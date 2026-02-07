# ファイルパス: scripts/training/train_conversational_snn.py
# 日本語タイトル: Conversational SNN Trainer (Stable RNN Version)

import sys
import os
import torch
import torch.nn as nn
import json
import logging
import random
from torch.optim import AdamW

# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 実験的モデルの代わりに、ここで安定版モデルを定義
class SpikingRNN(nn.Module):
    """
    SNNのレート符号化近似としてGRUを使用する安定版モデル。
    小規模データの学習に極めて強い。
    """
    def __init__(self, vocab_size, d_model=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (Batch, Seq)
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        logits = self.fc(out)
        return logits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ChatTrainer")

class SimpleTokenizer:
    def __init__(self):
        # 必要な文字を網羅的に定義
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
        else:
            ids = ids[:max_len]
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
    def __init__(self, vocab_size, device):
        super().__init__()
        self.device = device
        self.tokenizer = SimpleTokenizer()
        # 安定版モデルを使用
        self.core = SpikingRNN(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            num_layers=2
        ).to(device)
        
    def forward(self, x):
        return self.core(x)

    def generate(self, prompt: str) -> str:
        self.eval()
        input_ids = self.tokenizer.encode(prompt, max_len=64)
        
        # 入力部分の有効なトークンのみ抽出（パディング削除）
        valid_indices = (input_ids != self.tokenizer.pad_token_id)
        curr_ids = input_ids[valid_indices].unsqueeze(0).to(self.device)
        
        generated = []
        for _ in range(100):
            with torch.no_grad():
                logits = self.core(curr_ids)
                # Greedy Decoding (最も確率の高い文字を選ぶ)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                generated.append(next_token.item())
        
        return self.tokenizer.decode(torch.tensor(generated))

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training on {device}")
    
    # データセット作成
    raw_data = [
        {"input": "Hello", "answer": "Hi! I am DORA."},
        {"input": "Who are you?", "answer": "I am a neuromorphic AI."},
        {"input": "What is SNN?", "answer": "SNN is Spiking Neural Network."},
        {"input": "Good morning", "answer": "Good morning!"},
        {"input": "Goodbye", "answer": "See you later."},
        {"input": "How are you?", "answer": "I am functioning normally."},
        {"input": "Tell me a joke", "answer": "Why did the neural network cross the road? To get to the other side."},
        {"input": "What can you do?", "answer": "I can chat, learn, and sleep."},
        {"input": "Are you alive?", "answer": "I am a digital life form."},
    ]
    
    # データ拡張
    training_texts = []
    for item in raw_data:
        # プロンプト形式を統一
        text = f"Q: {item['input']}\nAnswer: {item['answer']}"
        training_texts.append(text)
    
    # 重み付け: データを複製してバッチサイズを稼ぐ
    training_texts = training_texts * 20 
    random.shuffle(training_texts)

    model = ConversationalSNN(vocab_size=100, device=device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    logger.info(f"🚀 Starting training on {len(training_texts)} samples...")
    
    model.train()
    epochs = 200 # 確実に収束させる
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # バッチ学習（簡易的に1サンプルずつ）
        for text in training_texts:
            optimizer.zero_grad()
            token_ids = model.tokenizer.encode(text, max_len=64).unsqueeze(0).to(device)
            
            logits = model(token_ids)
            
            # 次の文字を予測
            shift_logits = logits[:, :-1, :].contiguous().view(-1, model.tokenizer.vocab_size)
            shift_labels = token_ids[:, 1:].contiguous().view(-1)
            
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(training_texts)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # テスト生成
            if avg_loss < 0.1:
                test_prompt = "Q: Hello\nAnswer:"
                gen = model.generate(test_prompt)
                print(f"   Debug Gen: {gen}")

    # 保存
    os.makedirs("workspace/models", exist_ok=True)
    torch.save(model.state_dict(), "workspace/models/chat_snn.pth")
    logger.info("💾 Model saved.")

if __name__ == "__main__":
    main()