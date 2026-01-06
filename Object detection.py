"""
Training script with ResNetFPNWithUNet support
Fixed version with better parameters to avoid detection issues
"""

import torch
import os
import json
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, build_backbone
from detectron2.layers import ShapeSpec, Conv2d
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator
import matplotlib.pyplot as plt

# Diffusers for UNet/VAE
try:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("⚠️  Diffusers not available, will use standard training")
    DIFFUSERS_AVAILABLE = False

# Configuration
MAPPING_PATH = "/mnt/nfs/nina/merged_mapping.json"
UNET_PATH = "/mnt/nfs/nina/nina/visa_task/visa_task/unet"
VAE_PATH = "/mnt/nfs/nina/nina/visa_task/visa_task/vae"

# ============================================================
# Clear previous registration if exists
# ============================================================
if "ResNetFPNWithUNet" in BACKBONE_REGISTRY._obj_map:
    del BACKBONE_REGISTRY._obj_map["ResNetFPNWithUNet"]

# ============================================================
# Register custom backbone with fixes
# ============================================================
@BACKBONE_REGISTRY.register()
class ResNetFPNWithUNet(Backbone):
    """
    Custom backbone with ResNet-FPN + UNet features
    Fixed version with better fusion strategy
    """
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        # Build standard ResNet-FPN
        original_name = cfg.MODEL.BACKBONE.NAME
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        self.resnet_fpn = build_backbone(cfg, input_shape)
        cfg.MODEL.BACKBONE.NAME = original_name
        
        # Check if we should use UNet
        self.use_unet = cfg.MODEL.get("USE_UNET", False) and DIFFUSERS_AVAILABLE
        
        if self.use_unet:
            print("🔧 Initializing UNet/VAE components...")
            
            # Load VAE
            self.vae = None
            self.has_vae = False
            try:
                self.vae = AutoencoderKL.from_pretrained(VAE_PATH)
                self.vae.eval()
                for param in self.vae.parameters():
                    param.requires_grad = False
                self.has_vae = True
                print("✅ VAE loaded successfully")
            except Exception as e:
                print(f"⚠️  VAE failed to load: {e}")
            
            # Load UNet
            self.unet = None
            self.has_unet = False
            try:
                self.unet = UNet2DConditionModel.from_pretrained(UNET_PATH)
                self.unet.eval()
                for param in self.unet.parameters():
                    param.requires_grad = False
                self.has_unet = True
                print("✅ UNet loaded successfully")
            except Exception as e:
                print(f"⚠️  UNet failed to load: {e}")
                self.use_unet = False
            
            # UNet projection layers - improved architecture
            if self.use_unet and self.has_unet:
                # Better projection with BatchNorm for stability
                self.unet_proj = nn.Sequential(
                    Conv2d(4, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    Conv2d(128, 256, 1),
                    nn.BatchNorm2d(256)
                )
                
                # Lower initial fusion weight to prevent disruption
                self.fusion_weight = nn.Parameter(torch.tensor(0.05))  # Reduced from 0.1
                print("✅ UNet projection layers initialized")
        
        # Output features
        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {f: 256 for f in self._out_features}
        self._out_feature_strides = {
            "p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64
        }
    
    def forward(self, x):
        # Get ResNet-FPN features
        features = self.resnet_fpn(x)
        
        # Add UNet features if enabled
        if self.use_unet and self.has_unet and self.unet is not None:
            try:
                device = x.device
                batch_size = x.shape[0]
                
                # VAE encoding
                if self.has_vae and self.vae is not None:
                    with torch.no_grad():
                        # Normalize input for VAE
                        x_norm = (x / 127.5) - 1.0
                        latent = self.vae.encode(x_norm).latent_dist.sample()
                        z = latent * 0.18215
                else:
                    # Fallback: simple downsample
                    h, w = x.shape[2:4]
                    target_h, target_w = h // 8, w // 8
                    z = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    
                    # Ensure 4 channels
                    if z.shape[1] != 4:
                        # Use adaptive pooling or padding
                        if z.shape[1] > 4:
                            z = z[:, :4, :, :]
                        else:
                            padding = torch.zeros(batch_size, 4 - z.shape[1], target_h, target_w, device=device)
                            z = torch.cat([z, padding], dim=1)
                
                # UNet forward pass
                with torch.no_grad():
                    t = torch.zeros(batch_size, device=device, dtype=torch.long)
                    # Empty context for unconditional generation
                    context = torch.zeros((batch_size, 77, 768), device=device)
                    unet_out = self.unet(z, t, encoder_hidden_states=context).sample
                
                # Project UNet features
                unet_feat = self.unet_proj(unet_out)
                
                # Fuse with FPN features at each level
                for name in ["p2", "p3", "p4", "p5"]:
                    if name in features:
                        target_size = features[name].shape[2:]
                        unet_resized = F.interpolate(
                            unet_feat, 
                            size=target_size, 
                            mode='bilinear', 
                            align_corners=False
                        )
                        # Weighted fusion with clamped weight
                        weight = torch.clamp(self.fusion_weight, 0.0, 0.3)
                        features[name] = features[name] + weight * unet_resized
                
                # Keep p6 unchanged (too low resolution for UNet)
                
            except Exception as e:
                print(f"⚠️  UNet forward pass failed: {e}")
                # Return original features on failure
        
        return features
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

# ============================================================
# Dataset functions (unchanged)
# ============================================================
def get_bbox_only_dicts(mapping_file, indices=None):
    """Create dataset with only bounding boxes"""
    with open(mapping_file, 'r') as f:
        mappings = json.load(f)
    
    if indices is not None:
        mappings = [mappings[i] for i in indices]
    
    dataset_dicts = []
    
    for idx, mapping in enumerate(mappings):
        img_path = mapping["image"]
        mask_path = mapping["mask"]
        
        if not os.path.exists(img_path):
            continue
        
        # Read image dimensions
        try:
            img = Image.open(img_path)
            width, height = img.size
        except:
            continue
        
        record = {
            "file_name": img_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": []
        }
        
        # Read mask and extract bounding boxes
        if os.path.exists(mask_path):
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 10:  # Skip very small regions
                            continue
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        annotation = {
                            "bbox": [x, y, w, h],
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": 0,
                            "area": area,
                            "iscrowd": 0
                        }
                        record["annotations"].append(annotation)
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
        
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_datasets():
    """Register train and validation datasets"""
    # Clear previous registrations
    for d in ["bbox_train", "bbox_val"]:
        try:
            DatasetCatalog.remove(d)
            MetadataCatalog.remove(d)
        except:
            pass
    
    # Load mappings
    with open(MAPPING_PATH, 'r') as f:
        mappings = json.load(f)
    
    # 80/20 split
    split_idx = int(len(mappings) * 0.8)
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(mappings)))
    
    print(f"📊 Dataset split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Register datasets
    DatasetCatalog.register("bbox_train", lambda: get_bbox_only_dicts(MAPPING_PATH, train_indices))
    DatasetCatalog.register("bbox_val", lambda: get_bbox_only_dicts(MAPPING_PATH, val_indices))
    
    # Set metadata
    for d in ["bbox_train", "bbox_val"]:
        MetadataCatalog.get(d).set(thing_classes=["defect"])
        MetadataCatalog.get(d).set(thing_colors=[(255, 0, 0)])

# ============================================================
# Improved configuration
# ============================================================
def setup_cfg(use_custom_backbone=False, improved_params=True):
    """Setup configuration with improved parameters"""
    cfg = get_cfg()
    
    # Base config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Custom backbone
    if use_custom_backbone:
        cfg.MODEL.BACKBONE.NAME = "ResNetFPNWithUNet"
        cfg.MODEL.USE_UNET = True
        print("📌 Using ResNetFPNWithUNet backbone")
    
    # Dataset
    cfg.DATASETS.TRAIN = ("bbox_train",)
    cfg.DATASETS.TEST = ("bbox_val",)
    
    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Solver - improved parameters
    if improved_params:
        cfg.SOLVER.IMS_PER_BATCH = 4  # Increased batch size
        cfg.SOLVER.BASE_LR = 0.001    # Higher learning rate
        cfg.SOLVER.MAX_ITER = 8000    # More iterations
        cfg.SOLVER.STEPS = (5000, 7000)
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.WARMUP_METHOD = "linear"
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        
        # Better anchor sizes for defects
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        
        # More proposals
        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        
        # Data augmentation
        cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    else:
        # Original parameters
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 3000
        cfg.SOLVER.STEPS = (1000, 2000)
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # General settings
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_unet" if use_custom_backbone else "_standard"
    suffix += "_improved" if improved_params else ""
    cfg.OUTPUT_DIR = f"./bbox_output_{timestamp}{suffix}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

# ============================================================
# Custom trainer with evaluation
# ============================================================
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

# ============================================================
# Training function
# ============================================================
def train(use_custom_backbone=False, improved_params=True, resume=False):
    """Train the model with optional improvements"""
    backbone_type = "ResNetFPNWithUNet" if use_custom_backbone else "Standard ResNet-FPN"
    params_type = "improved" if improved_params else "original"
    
    print(f"🚀 Starting training")
    print(f"   Backbone: {backbone_type}")
    print(f"   Parameters: {params_type}")
    
    # Register datasets
    register_datasets()
    
    # Setup config
    cfg = setup_cfg(use_custom_backbone, improved_params)
    
    # Print configuration
    print(f"\n📊 Training configuration:")
    print(f"  - Model: Faster R-CNN")
    print(f"  - Backbone: {backbone_type}")
    print(f"  - Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  - Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  - Base LR: {cfg.SOLVER.BASE_LR}")
    print(f"  - Warmup: {cfg.SOLVER.WARMUP_ITERS if improved_params else 'None'}")
    print(f"  - Output: {cfg.OUTPUT_DIR}")
    
    # Create trainer
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    
    # Train
    print("\n📈 Training...")
    trainer.train()
    
    print(f"\n✅ Training complete! Model saved to: {cfg.OUTPUT_DIR}")
    
    # Return config for further use
    return cfg

# ============================================================
# Evaluation function
# ============================================================
def evaluate(model_path=None, use_custom_backbone=False):
    """Evaluate a trained model"""
    print("📊 Evaluating model...")
    
    # Register datasets
    register_datasets()
    
    # Setup config
    cfg = setup_cfg(use_custom_backbone, improved_params=True)
    
    # Model path
    if model_path is None:
        model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Evaluate
    evaluator = COCOEvaluator("bbox_val", output_dir=cfg.OUTPUT_DIR)
    
    from detectron2.evaluation import inference_on_dataset
    from detectron2.data import build_detection_test_loader
    
    val_loader = build_detection_test_loader(cfg, "bbox_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("\n📊 Evaluation results:")
    print(results)
    
    return results

# ============================================================
# Visualization function
# ============================================================
def visualize_predictions(model_path=None, use_custom_backbone=False, 
                         num_samples=5, threshold=0.5):
    """Visualize predictions"""
    print("🎨 Visualizing predictions...")
    
    # Register datasets
    register_datasets()
    
    # Setup config
    cfg = setup_cfg(use_custom_backbone, improved_params=True)
    
    if model_path is None:
        model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Get validation dataset
    dataset_dicts = DatasetCatalog.get("bbox_val")
    
    # Visualize
    viz_dir = os.path.join(os.path.dirname(model_path), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for idx, d in enumerate(dataset_dicts[:num_samples]):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        # Predictions
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("bbox_val"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Ground truth
        v_gt = Visualizer(img[:, :, ::-1], MetadataCatalog.get("bbox_val"), scale=1.0)
        out_gt = v_gt.draw_dataset_dict(d)
        
        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(out_gt.get_image())
        ax1.set_title(f"Ground Truth ({len(d['annotations'])} boxes)")
        ax1.axis('off')
        
        ax2.imshow(out.get_image())
        ax2.set_title(f"Predictions ({len(outputs['instances'])} detections)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"result_{idx:03d}.png"))
        plt.close()
        
        print(f"  Image {idx}: {len(outputs['instances'])} detections")
    
    print(f"✅ Saved visualizations to {viz_dir}")

# ============================================================
# Main function
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "viz"], default="train")
    parser.add_argument("--use-unet", action="store_true", help="Use ResNetFPNWithUNet")
    parser.add_argument("--use-original-params", action="store_true", 
                       help="Use original parameters instead of improved ones")
    parser.add_argument("--model-path", help="Path to model for eval/viz")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    
    args = parser.parse_args()
    
    improved_params = not args.use_original_params
    
    if args.mode == "train":
        cfg = train(use_custom_backbone=args.use_unet, 
                   improved_params=improved_params,
                   resume=args.resume)
        print(f"\n💡 To evaluate: python {__file__} --mode eval --model-path {cfg.OUTPUT_DIR}/model_final.pth" + 
              (" --use-unet" if args.use_unet else ""))
        
    elif args.mode == "eval":
        evaluate(model_path=args.model_path, use_custom_backbone=args.use_unet)
        
    elif args.mode == "viz":
        visualize_predictions(model_path=args.model_path, 
                            use_custom_backbone=args.use_unet,
                            threshold=args.threshold)

if __name__ == "__main__":
    main()
