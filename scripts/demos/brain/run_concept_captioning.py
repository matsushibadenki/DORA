# ファイルパス: scripts/demos/brain/run_concept_captioning.py
# 日本語タイトル: Multimodal Concept Learning (Type Fixed)
# 目的: mypyエラー "Tensor not callable" を # type: ignore で抑制。

import sys
import time
import logging
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from typing import cast

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConceptLearning")

TEXT_VOCAB = {
    "zero": 1000, "one": 1001, "two": 1002, "three": 1003, "four": 1004,
    "five": 1005, "six": 1006, "seven": 1007, "eight": 1008, "nine": 1009,
    "digit": 1010, "[PAD]": 1011
}
IDX_TO_TEXT = {v: k for k, v in TEXT_VOCAB.items()}

def text_to_tokens(label_idx: int, device: torch.device) -> torch.Tensor:
    digit_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][label_idx]
    tokens = [TEXT_VOCAB["digit"], TEXT_VOCAB[digit_word]]
    return torch.tensor(tokens, device=device).unsqueeze(0)

class ConceptTrainer:
    def __init__(self, brain: ArtificialBrain, device: torch.device):
        self.brain = brain
        self.device = device
        
        # [Fix] Type ignore added
        self.brain.reset_state() # type: ignore
        with torch.no_grad():
            dummy_input = torch.zeros(1, 10).long().to(device)
            dummy_out = self.brain(dummy_input)
            if dummy_out.dim() > 2:
                actual_output_dim = dummy_out.shape[-1]
            else:
                actual_output_dim = dummy_out.shape[-1]
        
        # [Fix] Type ignore added
        self.brain.reset_state() # type: ignore
        
        logger.info(f"🧠 Detected Brain Output Dimension: {actual_output_dim}")
        
        self.img_proj = nn.Linear(actual_output_dim, brain.d_model).to(device)
        self.txt_proj = nn.Linear(actual_output_dim, brain.d_model).to(device)
        self.classifier = nn.Linear(brain.d_model, 10).to(device)
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        self.optimizer = optim.Adam(
            list(self.brain.parameters()) + 
            list(self.img_proj.parameters()) + 
            list(self.txt_proj.parameters()) + 
            list(self.classifier.parameters()) +
            [self.temperature],
            lr=2e-4, 
            weight_decay=1e-5
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def contrastive_loss(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        logits = (image_features @ text_features.T) / torch.clamp(self.temperature, min=0.01)
        labels = torch.arange(logits.shape[0], device=self.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.brain.train()
        self.img_proj.train()
        self.txt_proj.train()
        self.classifier.train()
        
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Hybrid Learning]")
        for batch_idx, (images, labels) in enumerate(pbar):
            # [Fix] Type ignore added
            self.brain.reset_state() # type: ignore
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            # --- Image Path ---
            img_tokens = (images.view(batch_size, -1) * 255).long()
            img_tokens = torch.clamp(img_tokens, 0, 255)
            
            self.optimizer.zero_grad()
            
            img_out = self.brain(img_tokens)
            if img_out.dim() > 2:
                img_emb_raw = img_out.mean(dim=1)
            else:
                img_emb_raw = img_out
            
            img_emb = self.img_proj(img_emb_raw)
            
            # --- Text Path ---
            txt_tokens_batch = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)
            for i, lbl in enumerate(labels):
                digit_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][lbl.item()]
                txt_tokens_batch[i, 0] = TEXT_VOCAB["digit"]
                txt_tokens_batch[i, 1] = TEXT_VOCAB[digit_word]
            
            # [Fix] Type ignore added
            self.brain.reset_state() # type: ignore
            txt_out = self.brain(txt_tokens_batch)
            if txt_out.dim() > 2:
                txt_emb_raw = txt_out.mean(dim=1)
            else:
                txt_emb_raw = txt_out
                
            txt_emb = self.txt_proj(txt_emb_raw)
            
            loss_clip = self.contrastive_loss(img_emb, txt_emb)
            
            img_logits = self.classifier(img_emb)
            loss_img_cls = self.ce_loss(img_logits, labels)
            
            txt_logits = self.classifier(txt_emb)
            loss_txt_cls = self.ce_loss(txt_logits, labels)
            
            loss = loss_clip + (loss_img_cls + loss_txt_cls) * 0.5
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CLIP": f"{loss_clip.item():.2f}"})

        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        return total_loss / len(train_loader)

    def encode_image(self, images):
        self.brain.eval()
        self.img_proj.eval()
        with torch.no_grad():
            self.brain.reset_state() # type: ignore
            img_tokens = (images.view(images.size(0), -1) * 255).long()
            out = self.brain(img_tokens)
            if out.dim() > 2:
                emb = out.mean(dim=1)
            else:
                emb = out
            return self.img_proj(emb)

    def encode_text(self, text_tokens):
        self.brain.eval()
        self.txt_proj.eval()
        with torch.no_grad():
            self.brain.reset_state() # type: ignore
            out = self.brain(text_tokens)
            if out.dim() > 2:
                emb = out.mean(dim=1)
            else:
                emb = out
            return self.txt_proj(emb)

def run_concept_demo():
    print("\n" + "="*60)
    print("🧠 Phase 3: Multimodal Concept Learning (Hybrid Loss)")
    print("="*60 + "\n")

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    container.config.from_yaml(str(config_path))
    container.config.device.from_value("cpu")
    
    brain = cast(ArtificialBrain, container.artificial_brain())
    device = brain.device
    print(f"✅ Brain Initialized on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])
    
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_subset = torch.utils.data.Subset(full_dataset, range(5000))
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_subset = torch.utils.data.Subset(test_dataset, range(100))
    
    trainer = ConceptTrainer(brain, device)
    
    epochs = 5
    print("\n📖 Learning Concepts (Binding Images <-> Text)...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(train_loader, epoch)
        print(f"   Epoch {epoch}: Loss = {loss:.4f}")
        brain.sleep_cycle()

    print("\n🧪 Testing Concept Formation")
    
    print("\n[Test 1] Image-to-Text Retrieval")
    
    correct_retrieval = 0
    total_test = 0
    
    candidate_texts = []
    for i in range(10):
        t = text_to_tokens(i, device)
        candidate_texts.append(t)
    
    text_batch = torch.cat(candidate_texts, dim=0)
    candidate_embs = trainer.encode_text(text_batch)
    candidate_embs = F.normalize(candidate_embs, dim=1)
        
    for i in range(20):
        img, lbl = test_subset[i]
        img = img.to(device).unsqueeze(0)
        
        img_emb = trainer.encode_image(img)
        img_emb = F.normalize(img_emb, dim=1)
        
        sims = img_emb @ candidate_embs.T
        pred_idx = torch.argmax(sims).item()
        
        if i < 5:
            pred_word = IDX_TO_TEXT[TEXT_VOCAB[["zero","one","two","three","four","five","six","seven","eight","nine"][pred_idx]]]
            match_mark = "✅" if pred_idx == lbl else "❌"
            print(f"   Image: [{lbl}] -> Brain thinks: '{pred_word}' {match_mark}")
        
        if pred_idx == lbl:
            correct_retrieval += 1
        total_test += 1
            
    print(f"   📊 Accuracy: {correct_retrieval}/{total_test} ({(correct_retrieval/total_test)*100:.1f}%)")

    print("\n[Test 2] Text-to-Image Retrieval")
    
    query_label = 3
    query_text = text_to_tokens(query_label, device)
    q_emb = trainer.encode_text(query_text)
    q_emb = F.normalize(q_emb, dim=1)
    
    pool_imgs = []
    pool_labels = []
    for i in range(50):
        img, lbl = test_subset[i]
        pool_imgs.append(img)
        pool_labels.append(lbl)
    
    pool_imgs_tensor = torch.stack(pool_imgs).to(device)
    pool_embs = trainer.encode_image(pool_imgs_tensor)
    pool_embs = F.normalize(pool_embs, dim=1)
    
    sims = q_emb @ pool_embs.T
    topk_vals, topk_idxs = torch.topk(sims, k=3)
    
    print(f"   Query: 'digit {['zero','one','two','three'][query_label]}'")
    print("   Top 3 retrieved images:")
    for rank, idx in enumerate(topk_idxs[0]):
        retrieved_lbl = pool_labels[idx.item()]
        mark = "✅" if retrieved_lbl == query_label else "❌"
        print(f"     {rank+1}. Image of [{retrieved_lbl}] (Sim: {topk_vals[0][rank]:.3f}) {mark}")

    print("\n✅ Multimodal Concept Demonstration Completed.")

if __name__ == "__main__":
    run_concept_demo()