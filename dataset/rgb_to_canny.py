import os
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from fourm.vq.vqvae import DiVAE
import torch

def get_masks_full(rgb_path: str, mask_generator: SamAutomaticMaskGenerator) -> list:
    rgb_img = Image.open(rgb_path).convert("RGB")
    return mask_generator.generate(np.array(rgb_img)), rgb_img  # masks on 512x512

def rgb_to_canny_to_tokens(
    rgb_img: Image.Image, 
    masks_full: list,
    crop_settings: np.ndarray, 
    aug_idx: int, 
    tok_edge_fourm,
    device: str,
    low: int = 100,
    high: int = 200) -> np.ndarray | None:

    # Crop and resize image
    x1, y1, x2, y2, flip = crop_settings[aug_idx]
    cropped = TF.crop(rgb_img, int(y1), int(x1), int(y2 - y1), int(x2 - x1))
    resized = cropped.resize((256, 256), resample=Image.BILINEAR)
    if flip:
        resized = TF.hflip(resized)
        
    transformed_masks = []
    for ann in masks_full:
        seg = ann['segmentation']           # (512, 512) bool
        seg_crop = seg[y1:y2, x1:x2]       # crop
        seg_resized = cv2.resize(
            seg_crop.astype(np.uint8),
            (256, 256),
            interpolation=cv2.INTER_NEAREST # nearest for binary masks
        )
        if flip:
            seg_resized = cv2.flip(seg_resized, 1)
        transformed_masks.append({**ann, 'segmentation': seg_resized.astype(bool)})

    if not transformed_masks:
        return np.zeros(256, dtype=np.int32)

    # Canny edges from SAM
    sorted_anns = sorted(transformed_masks, key=(lambda x: x['area']), reverse=True)
    h, w = sorted_anns[0]['segmentation'].shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_uint8 = (m * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, low, high)
        binary[edges > 0] = 255

    # Tokenize
    TARGET_SIZE = 256
    edges_4m = cv2.resize(binary, (TARGET_SIZE, TARGET_SIZE))
    edges_4m = edges_4m.astype(np.float32) / 255.0
    edges_4m = edges_4m * 2.0 - 1.0
    edges_4m = torch.tensor(edges_4m).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = tok_edge_fourm.tokenize(edges_4m) # [1, 16, 16]
        
    return tokens.reshape(256).cpu().numpy().astype(np.int32)

    
def process_split(
    split: str,
    rgb_base: str,
    crop_base: str,
    out_base: str,
    mask_generator: SamAutomaticMaskGenerator,
    tok_edge_fourm,
    device: str,
    n_aug: int = 10):

    rgb_dir = os.path.join(rgb_base, split)
    crop_dir = os.path.join(crop_base, split, "crop_settings")
    out_dir = os.path.join(out_base, split, "tok_canny@256")
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith("_domain_rgb.png")])
    print(f"[{split}] Found {len(rgb_files)} RGB files.")

    for rgb_fname in rgb_files:
        parts = rgb_fname.replace("_domain_rgb.png", "").split("_")
        point_idx = int(parts[1])
        out_path = os.path.join(out_dir, f"{point_idx:05d}.npy")
        if os.path.exists(out_path):
            continue

        crop_path = os.path.join(crop_dir, f"{point_idx:05d}.npy")
        if not os.path.exists(crop_path):
            print(f"  Missing crop settings for {rgb_fname}, skipping.")
            continue

        crop_settings = np.load(crop_path)
        rgb_path = os.path.join(rgb_dir, rgb_fname)

        masks_full, rgb_img = get_masks_full(rgb_path, mask_generator)
        
        all_tokens = [] # will be 10 x [256]
        for aug_idx in range(n_aug):
            tokens = rgb_to_canny_to_tokens(
                rgb_img, masks_full, crop_settings, aug_idx, tok_edge_fourm, device)  # [256]
            
            all_tokens.append(tokens)

        np.save(out_path, np.stack(all_tokens, axis=0))
        print(f"  Done: {rgb_fname}")

        
if __name__ == "__main__":
    device = "cuda"
    sam_checkpoint = "/scratch/tmaillar/sam_vit_h_4b8939.pth"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=0.88,     
        stability_score_thresh=0.95,
        min_mask_region_area=5000,    # default 0 
        points_per_side=15,          # default 32
    )
    tok_edge_fourm = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_edge_8k_224-512').eval().to(device)

    RGB_path = "/scratch/tmaillar/rgb"
    CROP_path = "/work/com-304/datasets/clevr_com_304"
    CANNY_output_path = "/scratch/tmaillar/canny"

    # change to ["train", "val", "test"]
    process_split("train", RGB_path, CROP_path, CANNY_output_path, mask_generator, tok_edge_fourm, device)
    
