import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from models.FECNet import FECNet  # make sure this matches the repo structure
import pandas as pd
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FECNet(pretrained=True).to(device)
# Load FECNet model
def load_fecnet(model_path='pretrained_models/FECNet.pth'):
    model = FECNet(pretrained=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Initialize face detector
mtcnn = MTCNN(image_size=256, device=device)

# Compute expression embedding
def get_expression_embedding(image_path, model):
    image = Image.open(image_path).convert("RGB")
    print(f"🔍 Processing image: {image_path}")
    print(f"Image size: {image.size}")
    face = mtcnn(image)

    if face is None:
        print(f"❌ Face not detected in {image_path}")
        return None

    face = face.unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        embedding = model(face)  # shape: [1, 16]
    return embedding.squeeze(0)  # shape: [16]

# Compute L2 expression error
def compute_expression_error(path_swapped, path_target, model):
    emb_swapped = get_expression_embedding(path_swapped, model)
    emb_target = get_expression_embedding(path_target, model)

    if emb_swapped is None or emb_target is None:
        return None

    error = torch.norm(emb_swapped - emb_target, p=2).item()
    return error

# Compare folders
def compare_folders(folder1, folder2, output_csv="expression_error_results_ori.csv"):
    results = []
    common_files = sorted(set(os.listdir(folder1)) & set(os.listdir(folder2)))

    for fname in common_files:
        path1 = os.path.join(folder1, fname)
        path2 = os.path.join(folder2, fname)

        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg")):
            continue

        error = compute_expression_error(path1, path2, model)
        if error is not None:
            print(f"[OK] {fname}: Expression error = {error:.4f}")
            results.append({"filename": fname, "expression_error": error})
        else:
            print(f"[FAIL] {fname}: Error computing expression")

    df = pd.DataFrame(results)

    # Compute mean and add to end of CSV
    if not df.empty:
        mean_error = df["expression_error"].mean()
        print(f"\n📊 Mean Expression Error: {mean_error:.4f}")
        df.loc[len(df.index)] = ["MEAN", mean_error]  # Append row with mean

    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

# Main runner
if __name__ == "__main__":
    # Paths to your images
    
    folder_swapped = "/mnt/HDD6/thangql/cv_final_prj/DiffSwap_freeGuide/data/portrait/merged_images_original_diffswap"
    folder_target = "/mnt/HDD6/thangql/cv_final_prj/DiffSwap_freeGuide/data/portrait/align/target"

    compare_folders(folder_swapped, folder_target)