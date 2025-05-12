
## Environment setup
```
conda create -n DiffFace python=3.9.7
conda activate DiffFace

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch 

pip install -r requirements.txt
```


## Download Pretrained Weights
The weights required for the execution of our DiffFace can be downloaded from [link](
https://gisto365-my.sharepoint.com/:f:/g/personal/hongieee_gm_gist_ac_kr/Eolr4xhyDZdJhEqAjuVCMN8B7pdbvnMxEMiT4jB7w63uHg?e=KDoCnr). 
```
mkdir checkpoints
mv Arcface.tar checkpoints/ 
mv FaceParser.pth checkpoints/ 
mv GazeEstimator.pt checkpoints/
mv Model.pt checkpoints/
```

## Directories structure

The dataset and checkpoints should be placed in the following structures below

```
DiffFace
├── checkpoints
    ├── Arcface.tar
    ├── FaceParser.pth
    ├── GazeEstimator.pt
    └── Model.pt
├── data
    └── src
        ├── 001.png
        └── ...
    └── targ
        ├── 001.png
        └── ...
├── models
├── optimization
├── utils
└── main.py
```

## Inference 

Place source and target images in data/src, and data/targ. Then run the following. 

```
python main.py --output_path output/example
```


