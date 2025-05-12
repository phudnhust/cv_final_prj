# DiffSwap

## Installation
Please first install the environment following [stable-diffusion](https://github.com/CompVis/stable-diffusion)
```
conda env create -f env.yaml
conda activate ldm
pip install -r requirements.txt
```

Please download the checkpoints from [[here]](https://drive.google.com/drive/folders/1bEgYPzsXIcqfAvJOUb8Mkf2E7qrGh4oo?usp=drive_link), and put them under the  `checkpoints/` folder. 
The resulting file structure should be:

```
├── checkpoints
│   ├── diffswap.pth
│   ├── glint360k_r100.pth
│   ├── shape_predictor_68_face_landmarks.dat 
│   ├── FaceParser.pth 
│   ├── GazeEstimator.pt 
│   └── diffswap.pth
```

## Inference
Please put the source images and target images in `data/portrait_jpg` to inferent
1.  Default setting (Gaze and Frequency domain).
```
python pipeline.py
```
the swapped results are saved in `data/portrait/swap_res_ori`.


2. Gaze only
Remove --guide_fq argument in tests/face_swap.sh
```
python pipeline.py
```

2. Frequency Domain only
Remove --guide_gaze argument in tests/face_swap.sh
```
python pipeline.py
```



