# table_rec

## requirements

Since this program use Qwen2-VL, the `transformers` package must be newer than **4.44**. 

## create table images 

**step 1**

Place weight file `model_final.pth` under `assets/layout_models`.

The correct structure is shown as followed:

``` 
(mineru) root@10:/data-pfs/jd/programs/Table_Rec# tree 
.
├── README.md
├── assets
│   └── layout_models
│       ├── config.json
│       ├── layoutlmv3_base_inference.yaml
│       └── model_final.pth
├── scripts
│   └── extract_tables.py
└── utils
    ├── __init__.py
    ├── io.py
    ├── table_det.py
    └── table_explore.py
```

**step 2** 

`python scripts/extract_tables.py`

##  environment

Packages:
1. detectron2
2. numpy==1.25
3. loguru 
4. pillow==9.2


If you encounter the following problem (CUDA 11.3 with nvcc compiler was installed):

```
RuntimeError:
        The detected CUDA version (11.3) mismatches the version that was used to compile
        PyTorch (12.1). Please make sure to use the same CUDA versions.
```

run following commands to resolve it:

```
conda create -n det python=3.10 -y 
conda activate det
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install numpy==1.25
pip install magic-pdf[full]==0.7.0b1 --extra-index-url https://wheels.myhloli.com
apt-get update && apt-get install libgl1
```


