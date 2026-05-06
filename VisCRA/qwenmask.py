VISION_TOKEN = 151655
GRID_DIM = 28
GAUSSIAN_SIGMA = 14
ATTENTION_BLOCK_SIZE = 12
ATTENTION_LAYER = 18

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qwen_vl_utils import process_vision_info
from utils import read_mm_safebench ,process_hades_data
from transformers import MllamaForConditionalGeneration

torch.backends.cuda.enable_flash_sdp(True)  
torch.backends.cuda.enable_mem_efficient_sdp(True) 

def find_top3_blocks_with_stride(matrix, block_size=5, stride=3, max_overlap=40):
    integral = np.cumsum(np.cumsum(matrix, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
    
    rows, cols = matrix.shape
    candidates = []
    
    for i in range(0, rows - block_size + 1, stride):
        for j in range(0, cols - block_size + 1, stride):
            total = integral[i+block_size, j+block_size] - integral[i, j+block_size] - integral[i+block_size, j] + integral[i, j]
            candidates.append((-total, i, j))  
    
    candidates.sort()
    
    selected = []
    for cand in candidates:
        current_total = -cand[0]
        x, y = cand[1], cand[2]
        conflict = False
        
        for (s_i, s_j) in selected:
            x_overlap = max(0, min(x + block_size, s_i + block_size) - max(x, s_i))
            y_overlap = max(0, min(y + block_size, s_j + block_size) - max(y, s_j))
            overlap_area = x_overlap * y_overlap
            
            if overlap_area >= max_overlap:
                conflict = True
                break
        
        if not conflict:
            selected.append((x, y))
            if len(selected) >= 3:
                break  
    
    return selected[:3]

def visualize_attention(image_input, attention_tensor, block_size, output_dir, color="green"):
    os.makedirs(output_dir, exist_ok=True)
    
    original_path = os.path.join(output_dir, "original.png")
    image_input.save(original_path)
    
    img_width, img_height = image_input.size
    
    # -------------- LLAMA 3.2 VISION TILE FIX --------------
    if len(attention_tensor) == 6404:
        # Llama 3.2 Vision generates exactly 6404 cross-attention tokens 
        # (4 special tokens + up to 4 image tiles of 40x40 patches each)
        ratio = img_width / img_height
        if ratio > 2.5:   num_w, num_h = 4, 1  # Super wide
        elif ratio > 1.5: num_w, num_h = 2, 1  # Wide
        elif ratio < 0.4: num_w, num_h = 1, 4  # Super tall
        elif ratio < 0.7: num_w, num_h = 1, 2  # Tall
        else:             num_w, num_h = 2, 2  # Square-ish
        
        # Remove the 4 special tokens and grab the active tiles
        patches = attention_tensor[4:].reshape(4, 40, 40)
        active_tiles = patches[:num_h * num_w].reshape(num_h, num_w, 40, 40)
        
        # Merge the tiles into a single 2D heatmap grid
        attention_2d = active_tiles.transpose(0, 2, 1, 3).reshape(num_h * 40, num_w * 40)
        
        # Override the grid dimensions
        grid_h, grid_w = num_h * 40, num_w * 40
        
        # Calculate dynamic scales so the mask is drawn correctly on the original image
        scale_x = img_width / grid_w
        scale_y = img_height / grid_h
        
    else:
        # Original Qwen Logic Fallback
        grid_w = img_width // GRID_DIM  
        grid_h = img_height // GRID_DIM  
        attention_2d = attention_tensor[:grid_h * grid_w].reshape(grid_h, grid_w)
        scale_x = GRID_DIM
        scale_y = GRID_DIM
    # -------------------------------------------------------

    # Mask out the borders (to prevent edge artifacts from being flagged as hotspots)
    attention_2d[:4, :] = 0    
    attention_2d[-12:, :] = 0  
    attention_2d[:, 0] = 0     
    attention_2d[:, -1] = 0
    
    # Find the top hotspots
    top3_blocks = find_top3_blocks_with_stride(attention_2d, block_size=block_size, stride=3)
    
    mask_paths = []
    for idx, (block_row, block_col) in enumerate(top3_blocks):
        masked_img = image_input.copy()
        draw = ImageDraw.Draw(masked_img)
        
        # Use our dynamic scales instead of the hardcoded GRID_DIM
        mask_x = block_col * scale_x  
        mask_y = block_row * scale_y  
        mask_width = block_size * scale_x
        mask_height = block_size * scale_y
        
        mask_x_end = min(mask_x + mask_width, img_width)
        mask_y_end = min(mask_y + mask_height, img_height)
        
        draw.rectangle(
            (mask_x, mask_y, mask_x_end, mask_y_end),
            fill=color  
        )
        
        mask_path = os.path.join(output_dir, f"mask_block{idx}.png")
        masked_img.save(mask_path)
        mask_paths.append(mask_path)
    
    # Save Heatmap
    plt.figure(figsize=(img_width/100, img_height/100), dpi=100)
    plt.imshow(attention_2d, 
              cmap="viridis",
              aspect="auto",
              extent=[0, img_width, img_height, 0])  
    plt.axis("off")
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    # Save Overlay
    upsampled = attention_2d.repeat(int(max(1, scale_y)), axis=0).repeat(int(max(1, scale_x)), axis=1)
    upsampled = upsampled[:img_height, :img_width]
    smoothed = gaussian_filter(upsampled, sigma=GAUSSIAN_SIGMA)
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    
    plt.figure(figsize=(img_width/100, img_height/100), dpi=100)
    plt.imshow(image_input)
    plt.imshow(normalized, 
              cmap="viridis", 
              alpha=0.5,
              extent=[0, img_width, img_height, 0])
    plt.axis("off")
    overlay_path = os.path.join(output_dir, "overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    return {
        "original": original_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "masks": mask_paths
    }

def load_model(model_name): 
    if 'qwen' in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left", use_fast=True
        )
        return model, processor
    else:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

def prepare_inputs(model_name, processor, query: str, image: str):
    if 'qwen' in model_name.lower():
        placeholders = [{"type": "image", "image": image}] 
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": f"{query} Answer:"},
                ]
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        stop_token_ids = None
        return prompt, image_inputs, stop_token_ids
    else:
        image_obj = Image.open(image).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{query} Answer:"}
                ]
            }
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return prompt, image_obj

def generate(model, model_name, processor, query, image, device="cuda"):
    if 'qwen' in model_name.lower():
        text_inputs, image_inputs, stop_token_ids = prepare_inputs(model_name, processor, query, image)

        inputs = processor(
            text=[text_inputs],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        num_image_tokens = sum(inputs['input_ids'][0]==VISION_TOKEN)
        image_token_start= torch.where(inputs['input_ids'][0] == VISION_TOKEN)[0][0]
        image_token_end = torch.where(inputs['input_ids'][0] == VISION_TOKEN)[0][-1]

        with torch.no_grad():
            generated_ids = model(**inputs,output_attentions =True)
            output_attentions = generated_ids['attentions']
            att_layer = ATTENTION_LAYER
            block_size = ATTENTION_BLOCK_SIZE
            avg_attn = output_attentions[att_layer][0, :, -1, image_token_start:image_token_end+1].mean(0).float().cpu().numpy()
            parts = os.path.normpath(image).split(os.sep)
            if args.dataset == "HADES":
                green_output_dir = f"./attention_mask_{ATTENTION_BLOCK_SIZE}(green)/{args.dataset}/{parts[-3]}/{parts[-2]}"
            if args.dataset == "mm-safebench":
                index = os.path.splitext(os.path.basename(image))[0]
                green_output_dir = f"./attention_mask_{ATTENTION_BLOCK_SIZE}(green)/{args.dataset}/{parts[-3]}/{index}"
            

            visualize_attention(image_inputs[0], avg_attn, block_size, green_output_dir, color="green")
            
            return None
    else:
        text_inputs, image_obj = prepare_inputs(model_name, processor, query, image)

        inputs = processor(
            images=image_obj,
            text=text_inputs,
            return_tensors="pt"
        ).to(model.device)

        # 1. Prepare a list to catch our cross-attention weights
        captured_attentions = []

        # 2. Define the "wiretap" hook function
        def cross_attn_hook(module, input, output):
            # The cross attention module outputs: (attn_output, attn_weights, past_key_value)
            # We want output[1], which contains the raw attention weights
            if len(output) > 1 and output[1] is not None:
                captured_attentions.append(output[1].detach().cpu())

        # 3. Attach the hook to the exact cross-attention layer (e.g., Layer 18)
        target_layer = model.language_model.model.layers[ATTENTION_LAYER].cross_attn
        hook_handle = target_layer.register_forward_hook(cross_attn_hook)

        with torch.no_grad():
            # 4. Run the forward pass (forces the layer to calculate attentions)
            _ = model(**inputs, output_attentions=True)
            
        # 5. Remove the hook so it doesn't leak into the next loop iteration
        hook_handle.remove()

        # 6. Extract the captured cross-attention
        if not captured_attentions:
            print("Warning: Cross-attention weights were not captured.")
            return None
            
        cross_attentions = captured_attentions[0]
        
        # We want the attention from the *last* text token (-1) to all vision tokens,
        # averaged across all attention heads (.mean(0))
        avg_attn = cross_attentions[0, :, -1, :].mean(0).float().numpy()

        parts = os.path.normpath(image).split(os.sep)
        if args.dataset == "HADES":
            green_output_dir = f"./attention_mask_{ATTENTION_BLOCK_SIZE}(green)/{args.dataset}/{parts[-3]}/{parts[-2]}"
        if args.dataset == "mm-safebench":
            index = os.path.splitext(os.path.basename(image))[0]
            green_output_dir = f"./attention_mask_{ATTENTION_BLOCK_SIZE}(green)/{args.dataset}/{parts[-3]}/{index}"
        
        visualize_attention(image_obj, avg_attn, ATTENTION_BLOCK_SIZE, green_output_dir, color="green")
        
        return None



def main(args):
    model_path = {
        'qwen2_5-vl': 'pre_train_models/qwen2_5-vl-7b',
        'Llama-3.2-11b-V': 'pre_train_models/Llama-3.2-11b-Vision-Instruct'
    }
    model_name = model_path[args.model_name]
    model, processor = load_model(model_name)

    if args.dataset == "mm-safebench":
        dataset = read_mm_safebench(args.input_dir, img_type="SD_TYPO",query_type = "Rephrased Question(SD)")
        for item in tqdm(dataset, desc="Processing queries"):
            query = item["ori_harm_query"]
            image = item["image"]
            response = generate(model, model_name, processor, query, image)
            torch.cuda.empty_cache()

    if args.dataset == "HADES":
        dataset = process_hades_data(args.input_dir)
        for item in tqdm(dataset, desc="Processing queries"):
            query = item["ori_harm_query"]
            image = item["image"]
            response = generate(model, model_name, processor, query, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2_5-vl', help='Model name to use for inference')
    parser.add_argument('--dataset', default="HADES", help='dataset to evaluation')
    parser.add_argument('--input_dir', default="datasets/Hades", help='Path to input file')
    args = parser.parse_args()
    print(args)
    main(args)
