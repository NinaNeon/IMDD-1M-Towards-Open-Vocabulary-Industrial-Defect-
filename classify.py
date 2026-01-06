import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor

# ================== 設定 ==================
JSON_PATH = "/Users/nina/Documents/cat500/src_0319_categories.json"
IMAGE_BASE_DIR = "/Users/nina/Documents/0319/"
UNET_PATH = "/Users/nina/Documents/cat500/unet"
VAE_PATH = "/Users/nina/Documents/cat500/vae"

# ================== Dataset ==================
class DefectDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.labels = [item['text'].split(',')[-1].strip() for item in self.data]
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['file_name']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_encoder.transform([item['text'].split(',')[-1].strip()])[0]
        return img, label

# ================== Encoder & Classifier ==================
class UNetEncoder(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding

        # 凍結所有參數
        for p in unet.parameters():
            p.requires_grad = False

    def forward(self, latent, timestep):
        # 時間步嵌入
        t = torch.tensor([timestep], device=latent.device)
        temb = self.time_proj(t)
        temb = self.time_embedding(temb)

        # 初始卷積
        h = self.conv_in(latent)
        
        # 準備空的 encoder_hidden_states (與批次大小匹配)
        encoder_hidden_states = torch.zeros((latent.shape[0], 77, 768), device=latent.device)

        # 通過所有下採樣塊
        for block in self.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                h, _ = block(h, temb, encoder_hidden_states=encoder_hidden_states)
            else:
                h, _ = block(h, temb)
                
        return h

class DefectClassifier(nn.Module):
    def __init__(self, unet, num_classes):
        super().__init__()
        self.encoder = UNetEncoder(unet)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),  # 最後一層 down block 的輸出是 (B, 1280, 8, 8)
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加 dropout 以減少過擬合
            nn.Linear(512, num_classes)
        )

    def forward(self, x, t=0):
        feat = self.encoder(x, t)
        feat = self.pool(feat)
        return self.classifier(feat)

# ================== Training ==================
def train():
    with open(JSON_PATH) as f:
        data = json.load(f)
    
    # 修正資料路徑 (如果需要)
    for item in data:
        if not item['file_name'].startswith(IMAGE_BASE_DIR):
            item['file_name'] = IMAGE_BASE_DIR + item['file_name'].split('/')[-1]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = DefectDataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 加載 VAE
    print("加載 VAE 模型...")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
    vae.eval()

    # 加載 UNet
    print("加載 UNet 模型...")
    unet = UNet2DConditionModel.from_pretrained(UNET_PATH).to(device)
    # 設置簡單的注意力處理器
    unet.set_attn_processor(AttnProcessor())
    
    # 建立分類器
    num_classes = len(dataset.label_encoder.classes_)
    print(f"類別數量: {num_classes}")
    print(f"類別映射: {dict(zip(dataset.label_encoder.classes_, range(num_classes)))}")
    
    model = DefectClassifier(unet, num_classes=num_classes).to(device)

    # 訓練設置
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 訓練循環
    num_epochs = 10
    best_loss = float('inf')
    
    print("開始訓練...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 使用 VAE 將圖像轉換為潛在表示
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
            
            # 前向傳播
            outputs = model(latents, t=0)
            loss = criterion(outputs, labels)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 統計
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 顯示批次進度
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        # 計算平均損失和準確率
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} 完成 | Avg Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        
        # 更新學習率調度器
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'class_mapping': {c: i for i, c in enumerate(dataset.label_encoder.classes_)}
            }, "best_defect_classifier.pth")
            print(f"保存最佳模型於 epoch {epoch+1}, loss: {best_loss:.4f}")

    print("訓練完成!")
    torch.save(model.state_dict(), "final_defect_classifier.pth")

if __name__ == "__main__":
    train()
