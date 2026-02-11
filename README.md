
# AV-DAR: Differentiable Room Acoustic Rendering with Multi-View Vision Priors

### [Project Page](https://humathe.github.io/avdar/)  | [Paper(arXiv)](https://arxiv.org/pdf/2504.21847) 

> Official implementation of the ICCV 2025 Oral paper _"Differentiable Room Acoustic Rendering with Multi-View Vision Priors."_

## Updates
- Oct 17, 2025: Released our training & evaluation code.

## Installation
### Environment
- Tested on Python 3.10, with Torch version '2.4.1+cu118', other version should be fine.
- Install dependencies
```bash
git clone https://github.com/HuMathe/av-dar.git
cd av-dar
conda create -n av-dar python=3.10 -y
conda activate av-dar
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```


### Datasets:
Download [Hearing Anything Anywhere](https://masonlwang.com/hearinganythinganywhere/) and [Real Acoustic Field](https://facebookresearch.github.io/real-acoustic-fields/) based on their instruction.

Please download the datasets following their official instructions:
- [Hearing Anything Anywhere(HAA)](https://masonlwang.com/hearinganythinganywhere/)
- [Real Acoustic Field (RAF)](https://facebookresearch.github.io/real-acoustic-fields/)

Then update two lines in `config/base.yaml`:
```yaml
haa_data_dir: /path/to/HAA
raf_data_dir: /path/to/RAF/archive
```

### Preprocessed vision features
We use precomputed multi-view image features, and unproject them to the room's sample points. By default we expect them under:
```
preprocess/image-features/{haa,raf}/...
```
For EmptyRoom and FurnishedRoom, unzip the features.npy.zip files before training:
```
unzip features.npy.zip
```

Preprocessing scripts are provided:
```
preprocess/preprocess-haa.py
preprocess/preprocess-raf.py
```

**Note on HAA visual priors:** Images are rendered from textured meshes (not real RGB images).

We thank the authors of *Hearing Anything Anywhere* for sharing the textured meshes for the four scenes.

| Item | Path |
|---|---|
| Meshes | `preprocess/haa-visual/glb/<scene>.glb` |
| Rendered images + cameras | `preprocess/haa-visual/<scene>` |
| Image features | `preprocess/image-features/haa/<scene>` |

If you use the meshes, please cite: [Hearing Anything Anywhere](https://masonlwang.com/hearinganythinganywhere/)

## Usage
### Train (Hydra)
We use Hydra to manage configs:
```bash
# HAA — ClassroomBase @ 16 kHz
python train.py dataset=classroomBase-16K train=HAA-ClassroomBase-16K device=cuda:0

# Other HAA room types
# dataset=complexBase-16K | dampenedBase-16K | hallwayBase-16K
# train=HAA-ComplexBase-16K | HAA-DampenedBase-16K | HAA-HallwayBase-16K
```

RAF examples (16 kHz, different sparsity splits in configs):
```bash
# Empty room
python train.py dataset=EmptyRoom-16K-0.1% train=RAF-Empty-16K-0.1% device=cuda:0
python train.py dataset=EmptyRoom-16K-1% train=RAF-Empty-16K-1% device=cuda:0

# Furnished room (0.1%)
python train.py dataset=FurnishedRoom-16K-0.1% train=RAF-Furnished-16K-0.1% device=cuda:0
python train.py dataset=FurnishedRoom-16K-1% train=RAF-Furnished-16K-1% device=cuda:0
```
> Tip: `HYDRA_FULL_ERROR=1` helps with debugging config merges.

### Evaluate
```
# evaluate a trained run directory
python evaluate.py --config_dir /path/to/your/training/run
```


## Repository Structure
```bash
|-- av-dar
|   |-- core/   # io/run/typing...
|   |-- data/   # dataset loaders
|   |-- geometry/ # beam tracer
|   |-- model/  # renderer & sub-components...
|   `-- utils/
|-- config
|   |-- base.yaml
|   |-- dataset/ # data configs
|   `-- train/  # training configs (including model configs)
|-- data-split/ # datasplit json files
|-- evaluate.py
|-- mesh/*.obj  # room geometries for beam tracing
|-- preprocess/ # preprocess image features...
|-- README.md
`-- train.py
```



## TODOs
- Release the checkpoints for trained models.

 
## Data Attribution & Licenses

- **RAF-derived meshes** → CC BY-NC 4.0 (non-commercial). See details and change notes in `ATTRIBUTION.md`.  
- **HAA-derived meshes** (format conversion to `.obj` only) → CC BY 4.0. See details in `ATTRIBUTION.md`.

No endorsement is implied by the original authors or licensors.

## License (Code)
This repository’s code is released under the MIT License. See LICENSE.

## Citation
If you think this work is useful, please cite our paper.
```
@InProceedings{Jin_2025_ICCV,
    author    = {Jin, Derong and Gao, Ruohan},
    title     = {Differentiable Room Acoustic Rendering with Multi-View Vision Priors},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {37-47}
}
```

