# !pip install deepface
from deepface import DeepFace
from pathlib import Path
import numpy as np
import pandas as pd

original_diffswap_output_dir = Path('/mnt/HDD2/phudoan/my_stuff/DiffSwap/data/portrait/swap_res/diffswap_0.01')
modified_diffswap_output_dir = '/mnt/HDD2/phudoan/my_stuff/DiffSwap/modified_diffswap_output/'
source_dir = '/mnt/HDD2/phudoan/my_stuff/DiffSwap/data/portrait/source/'

img_num = []
original_output = []
modified_output = []

for original_output_img_path in original_diffswap_output_dir.rglob('*.png'):
    filename_without_ext = str(original_output_img_path).split('/')[-2]
    src_img_path = source_dir + filename_without_ext + '.png'
    modified_output_img_path = modified_diffswap_output_dir + '0' + filename_without_ext + '.png'

    metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]

    try:
        detect_face_source = DeepFace.extract_faces(src_img_path, detector_backend='retinaface')
        detect_face_original = DeepFace.extract_faces(original_output_img_path, detector_backend='retinaface')
        detect_face_modified = DeepFace.extract_faces(modified_output_img_path, detector_backend='retinaface')
    except ValueError as e:
        print("Caught a ValueError:", filename_without_ext)
        continue

    original_diffswap_result = DeepFace.verify(
        img1_path = str(original_output_img_path), img2_path = src_img_path, distance_metric = metrics[0], detector_backend='retinaface'
    )['distance']

    modified_diffswap_result = DeepFace.verify(
        img1_path = modified_output_img_path, img2_path = src_img_path, distance_metric = metrics[0], detector_backend='retinaface'
    )['distance']

    print(f'Similarity {filename_without_ext} original diffswap: {original_diffswap_result}  modified diffswap: {modified_diffswap_result}')

    img_num.append(filename_without_ext)
    original_output.append(original_diffswap_result)
    modified_output.append(modified_diffswap_result)

df = pd.DataFrame({
    'img_num': img_num,
    'original_output': original_output,
    'modified_output': modified_output
}).sort_values(by='img_num')

df.to_csv('face_embedding_result.csv', index=False)
