# ==========================================
# 整合自訂 UNet/VAE 到 ODISE
# 保存為: integrate_custom_models.py
# ==========================================

import os
import sys

# 步驟 1: 修改 ODISE 主文件，載入您的 UNet/VAE
print("🔧 步驟 1: 修改 ODISE 主架構...")

odise_file = "/mnt/c/Users/USER/Desktop/ODISE/odise/modeling/meta_arch/odise.py"

# 讀取原始檔案
with open(odise_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 備份
backup_file = odise_file + ".backup"
if not os.path.exists(backup_file):
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已備份原始檔案: {backup_file}")

# 找到 __init__ 方法並在開頭插入自訂模型載入代碼
custom_loading_code = '''
        # ========== 載入自訂 UNet/VAE ==========
        print("🔄 載入自訂 UNet/VAE...")
        from diffusers import UNet2DConditionModel, AutoencoderKL
        
        custom_unet_path = r"/mnt/c/Users/USER/Desktop/mvtec_model/unet"
        custom_vae_path = r"/mnt/c/Users/USER/Desktop/mvtec_model/vae"
        
        try:
            self.custom_unet = UNet2DConditionModel.from_pretrained(custom_unet_path)
            self.custom_vae = AutoencoderKL.from_pretrained(custom_vae_path)
            
            # 凍結參數
            for param in self.custom_unet.parameters():
                param.requires_grad = False
            for param in self.custom_vae.parameters():
                param.requires_grad = False
            
            print(f"✅ 自訂 UNet 已載入並凍結")
            print(f"✅ 自訂 VAE 已載入並凍結")
            
            # 替換原本的 diffusion model
            if hasattr(self, 'diffusion'):
                self.diffusion.unet = self.custom_unet
                self.diffusion.vae = self.custom_vae
                print("✅ 已替換 diffusion model")
                
        except Exception as e:
            print(f"⚠️  載入自訂模型失敗: {e}")
            print("   將使用原本的模型")
        # =====================================
'''

# 找到 super().__init__() 之後插入
if "super().__init__()" in content:
    content = content.replace(
        "super().__init__()",
        "super().__init__()" + custom_loading_code,
        1  # 只替換第一個
    )
    
    # 寫回檔案
    with open(odise_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已修改 {odise_file}")
else:
    print("⚠️  找不到 super().__init__()，需要手動修改")

# 步驟 2: 創建測試腳本
print("\n🔧 步驟 2: 創建測試腳本...")

test_script = '''#!/usr/bin/env python3
# 測試自訂 UNet/VAE
import sys
sys.path.insert(0, '/mnt/c/Users/USER/Desktop/ODISE')

import torch
from detectron2.config import LazyConfig
from odise.modeling.meta_arch.odise import ODISE

print("載入配置...")
cfg = LazyConfig.load("configs/Panoptic/odise_label_coco_min_debug.py")
cfg.train.device = "cpu"

print("建立模型（會自動載入自訂 UNet/VAE）...")
model = ODISE(cfg).to("cpu")

print("\\n✅ 模型建立完成！")
print("檢查是否有 custom_unet 和 custom_vae 屬性...")

if hasattr(model, 'custom_unet'):
    print("✅ 找到 custom_unet")
else:
    print("❌ 沒有 custom_unet")

if hasattr(model, 'custom_vae'):
    print("✅ 找到 custom_vae")
else:
    print("❌ 沒有 custom_vae")
'''

test_script_path = "/mnt/c/Users/USER/Desktop/ODISE/test_custom_models.py"
with open(test_script_path, 'w', encoding='utf-8') as f:
    f.write(test_script)

os.chmod(test_script_path, 0o755)
print(f"✅ 測試腳本已創建: {test_script_path}")

# 步驟 3: 創建完整訓練腳本
print("\n🔧 步驟 3: 創建完整訓練腳本...")

train_script = '''#!/bin/bash
# 使用自訂 UNet/VAE 訓練

cd /mnt/c/Users/USER/Desktop/ODISE
export CUDA_VISIBLE_DEVICES=""

echo "開始訓練（使用自訂 UNet/VAE）..."

python tools/train_net.py \\
    --config-file configs/Panoptic/odise_label_coco_min_debug.py \\
    --num-gpus 0 \\
    train.device=cpu \\
    train.max_iter=20 \\
    train.checkpointer.period=5 \\
    dataloader.train.total_batch_size=1 \\
    dataset.train=mvtec_train \\
    dataset.test=mvtec_train \\
    train.output_dir=./output/mvtec_custom_unet

echo "訓練完成！模型保存在 ./output/mvtec_custom_unet/"
'''

train_script_path = "/mnt/c/Users/USER/Desktop/ODISE/train_with_custom_unet.sh"
with open(train_script_path, 'w', encoding='utf-8') as f:
    f.write(train_script)

os.chmod(train_script_path, 0o755)
print(f"✅ 訓練腳本已創建: {train_script_path}")

# 步驟 4: 創建推論腳本
print("\n🔧 步驟 4: 創建推論腳本...")

inference_script = '''#!/usr/bin/env python3
# 使用自訂 UNet/VAE 進行推論
import sys
sys.path.insert(0, '/mnt/c/Users/USER/Desktop/ODISE')

import torch
import cv2
import numpy as np
from detectron2.config import LazyConfig
from detectron2.checkpoint import DetectionCheckpointer
from odise.modeling.meta_arch.odise import ODISE

# 路徑
TEST_IMG = "/mnt/c/Users/USER/Downloads/mvtec_coco_format/1722.png"
GT_IMG = "/mnt/c/Users/USER/Downloads/mvtec_coco_format/172GT.png"
MODEL_PATH = "./output/mvtec_custom_unet/model_0000020.pth"  # 或最新的 checkpoint
OUTPUT = "/mnt/c/Users/USER/Downloads/mvtec_coco_format/result_custom.png"

print("載入模型...")
cfg = LazyConfig.load("configs/Panoptic/odise_label_coco_min_debug.py")
cfg.train.device = "cpu"

model = ODISE(cfg).to("cpu")
model.eval()

checkpointer = DetectionCheckpointer(model)
checkpointer.load(MODEL_PATH)
print("✅ 模型已載入（使用自訂 UNet/VAE）")

# 讀取圖片
print("\\n讀取圖片...")
img = cv2.imread(TEST_IMG)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gt = cv2.imread(GT_IMG, cv2.IMREAD_GRAYSCALE)

print(f"測試圖: {img.shape}")
print(f"GT: {gt.shape}")

# 推論
print("\\n執行推論...")
with torch.no_grad():
    height, width = img.shape[:2]
    image_torch = torch.as_tensor(img_rgb.copy()).permute(2, 0, 1)
    inputs = [{"image": image_torch, "height": height, "width": width}]
    
    outputs = model(inputs)

print("✅ 推論完成")

# 提取預測遮罩
if "instances" in outputs[0]:
    instances = outputs[0]["instances"]
    masks = instances.pred_masks.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    
    print(f"\\n檢測到 {len(masks)} 個區域")
    
    if len(masks) > 0:
        # 合併遮罩
        combined = np.zeros((height, width), dtype=np.uint8)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score > 0.3:  # 過濾低分
                combined = np.logical_or(combined, mask[:height, :width])
                print(f"  區域 {i+1}: 信心度 {score:.3f}")
        
        pred_mask = (combined * 255).astype(np.uint8)
        
        # 計算 IoU
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        iou = intersection / union if union > 0 else 0
        
        print(f"\\n📊 IoU: {iou:.4f}")
        
        # 保存結果
        cv2.imwrite(OUTPUT.replace(".png", "_mask.png"), pred_mask)
        
        # 創建對比圖
        comparison = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        comparison[pred_mask > 127] = [255, 0, 0]  # 預測=紅色
        cv2.imwrite(OUTPUT, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        print(f"✅ 結果已保存: {OUTPUT}")
    else:
        print("⚠️  沒有檢測到任何區域")
else:
    print("⚠️  輸出中沒有 instances")
'''

inference_script_path = "/mnt/c/Users/USER/Desktop/ODISE/inference_custom.py"
with open(inference_script_path, 'w', encoding='utf-8') as f:
    f.write(inference_script)

os.chmod(inference_script_path, 0o755)
print(f"✅ 推論腳本已創建: {inference_script_path}")

print("\n" + "="*60)
print("✅ 整合完成！")
print("="*60)
print("""
接下來的步驟：

1. 測試自訂模型是否載入：
   cd /mnt/c/Users/USER/Desktop/ODISE
   python test_custom_models.py

2. 訓練（使用自訂 UNet/VAE）：
   cd /mnt/c/Users/USER/Desktop/ODISE
   bash train_with_custom_unet.sh

3. 推論：
   cd /mnt/c/Users/USER/Desktop/ODISE
   python inference_custom.py

注意：訓練時會在開頭看到 "🔄 載入自訂 UNet/VAE..." 的訊息
""")
