import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

# Load pre-trained face detector and landmark predictor

def detect_face_pose(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)

    # Detect face and landmarks
    mtcnn = MTCNN(keep_all=False)
    box, prob, landmarks = mtcnn.detect(img_pil, landmarks=True)

    if landmarks is not None:
        lm = landmarks[0]  # Use the first face
        image_points = np.array([
            lm[0],  # left eye
            lm[1],  # right eye
            lm[2],  # nose
            lm[3],  # mouth left
            lm[4],  # mouth right
        ], dtype="double")

        # 3D model points (approximate coordinates in mm)
        model_points = np.array([
            [-30.0, 0.0, -30.0],   # left eye
            [30.0, 0.0, -30.0],    # right eye
            [0.0, 0.0, 0.0],       # nose
            [-20.0, -30.0, -30.0], # mouth left
            [20.0, -30.0, -30.0],  # mouth right
        ])

        # Camera intrinsics (approximate, adjust for real camera)
        h, w = image_rgb.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=model_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )


        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw, pitch, roll = angles  # In degrees
        print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")
        return [yaw, pitch, roll]
    else:
        print("No face detected.")

def calculate_pose_similarity_degree(pose1, pose2):
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    """
    # Define poses in degrees
    # pose1 = [yaw1, pitch1, roll1]
    # pose2 = [yaw2, pitch2, roll2]
    """

    # # Convert to rotation objects
    # r1 = R.from_euler('yxz', pose1, degrees=True)
    # r2 = R.from_euler('yxz', pose2, degrees=True)

    # # Compute angular difference (in radians)
    # angle_diff = r1.inv() * r2
    # angle_degrees = angle_diff.magnitude() * (180 / np.pi)

    # # print(f"Angular distance: {angle_degrees:.2f} degrees")
    # return angle_degrees

    return np.sqrt(
        (pose1[0] - pose2[0]) ** 2 +
        (pose1[1] - pose2[1]) ** 2 +
        (pose1[2] - pose2[2]) ** 2
    )

# !pip install deepface
from deepface import DeepFace
from pathlib import Path
import numpy as np
import pandas as pd

original_diffswap_output_dir = Path('/mnt/HDD2/phudoan/my_stuff/DiffSwap/data/portrait/swap_res/diffswap_0.01')
modified_diffswap_output_dir = '/mnt/HDD2/phudoan/my_stuff/DiffSwap/modified_diffswap_output/'
source_dir = '/mnt/HDD2/phudoan/my_stuff/DiffSwap/data/portrait/target/'

img_num = []
original_output = []
modified_output = []

for original_output_img_path in original_diffswap_output_dir.rglob('*.png'):
    filename_without_ext = str(original_output_img_path).split('/')[-2]
    src_img_path = source_dir + filename_without_ext + '.png'
    modified_output_img_path = modified_diffswap_output_dir + '0' + filename_without_ext + '.png'

    src_image_pose = detect_face_pose(src_img_path)
    original_image_pose = detect_face_pose(original_output_img_path)
    modified_image_pose = detect_face_pose(modified_output_img_path)

    original_diffswap_result = calculate_pose_similarity_degree(src_image_pose, original_image_pose)
    modified_diffswap_result = calculate_pose_similarity_degree(src_image_pose, modified_image_pose)

    print(f'Similarity {filename_without_ext} original diffswap: {original_diffswap_result}  modified diffswap: {modified_diffswap_result}')

    img_num.append(filename_without_ext)
    original_output.append(original_diffswap_result)
    modified_output.append(modified_diffswap_result)

df = pd.DataFrame({
    'img_num': img_num,
    'original_output': original_output,
    'modified_output': modified_output
}).sort_values(by='img_num')

df.to_csv('face_pose_result.csv', index=False)
