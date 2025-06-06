import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ----- Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ----- Inception Feature Extractor -----
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(pretrained=True, aux_logits=True)
        self.features = torch.nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            model.avgpool,
            torch.nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

# ----- Compute Feature Distance -----
def get_feature(img_path, model, device):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)
    return feat.squeeze(0).cpu().numpy()

def compute_pairwise_feature_distance(folder1, folder2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionFeatureExtractor().to(device).eval()

    img1_list = sorted(os.listdir(folder1))
    img2_list = sorted(os.listdir(folder2))

    assert len(img1_list) == len(img2_list), "Hai thư mục phải có cùng số lượng ảnh"

    distances = []
    for img1, img2 in tqdm(zip(img1_list, img2_list), total=len(img1_list)):
        path1 = os.path.join(folder1, img1)
        path2 = os.path.join(folder2, img2)
        feat1 = get_feature(path1, model, device)
        feat2 = get_feature(path2, model, device)
        dist = np.linalg.norm(feat1 - feat2)
        distances.append(dist)

    mean_distance = np.mean(distances)
    return mean_distance, distances

# ----- Example -----
# Đặt đường dẫn tới 2 thư mục ảnh:
source_folder = "/mnt/HDD6/thangql/cv_final_prj/DiffSwap_freeGuide/data/portrait/align/target"
swap_folder = "/mnt/HDD6/thangql/cv_final_prj/DiffSwap_freeGuide/data/portrait/merged_images"

mean_dist, pairwise_dists = compute_pairwise_feature_distance(swap_folder, source_folder)
print(f"\nMean Pairwise Feature Distance (FID-style): {mean_dist:.4f}")
