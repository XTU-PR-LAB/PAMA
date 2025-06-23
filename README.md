# Knowledge-embedded Transformer for 3D Human Pose Estimation
## Installation instructions

- Python 3.8
```
conda create --no-default-packages -n smpl python=3.8
conda activate smpl
```

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.12.0
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

- [opendr](https://github.com/polmorenoc/opendr)
```
pip install "git+https://github.com/polmorenoc/opendr.git#subdirectory=opendr"
```

- [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)
```
pip install "git+https://github.com/daniilidis-group/neural_renderer"
```

- other packages listed in `requirements.txt`
```
pip install -r requirements.txt
```

### necessary files

> mesh_downsampling.npz & DensePose UV data

- Run the following script to fetch mesh_downsampling.npz & DensePose UV data from other repositories.

```
bash fetch_data.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Fetch final_fits data from [SPIN](https://github.com/nkolot/SPIN#final-fits). [important note: using [EFT](https://github.com/facebookresearch/eft) fits for training is much better. Compatible npz files are available [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)]


## Run demo code
```
python3 demo.py --checkpoint=data/pretrained_model/model_checkpoint.pt --img_file examples/COCO_val2014_000000019667.jpg
```

## Run evaluation code

### 3DPW

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
# Example usage:
# 3DPW
python3 eval.py --checkpoint=data/pretrained_model/model_checkpoint.pt --dataset=3dpw --log_freq=20
```

## Run training code

**To perform training, we need to collect preprocessed files of training datasets first. This paper uses [EFT](https://github.com/facebookresearch/eft) fits. Compatible data is available [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5).**

The preprocessed labels have the same format as SPIN and can be retrieved from [here](https://github.com/nkolot/SPIN#fetch-data). Please refer to [SPIN](https://github.com/nkolot/SPIN) for more details about data preprocessing.

Similar to PyMAF, this model is trained on Human3.6M at the first stage and then trained on the mixture of both 2D and 3D datasets at the second stage. Example usage:
```
# training on COCO
CUDA_VISIBLE_DEVICES=0 python3 train.py --regressor pymaf_net --single_dataset --misc TRAIN.BATCH_SIZE 64
# training on mixed datasets
CUDA_VISIBLE_DEVICES=0 python3 train.py --regressor pymaf_net --pretrained_checkpoint path/to/checkpoint_file.pt --misc TRAIN.BATCH_SIZE 64
```
Running the above commands will use Human3.6M or mixed datasets for training, respectively. We can monitor the training process by setting up a TensorBoard at the directory `./logs`.


## Acknowledgments

The code is developed upon the following projects. Many thanks to their contributions.
- [PyMAF](https://github.com/HongwenZhang/PyMAF/)

- [SPIN](https://github.com/nkolot/SPIN)

- [VIBE](https://github.com/mkocabas/VIBE)

- [PIFu](https://github.com/shunsukesaito/PIFu)

- [DensePose](https://github.com/facebookresearch/DensePose)

- [HMR](https://github.com/akanazawa/hmr)

- [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch)



### Citation
If you find our code or paper helps, please consider citing:
```
@article{
  title={Knowledge-Embedded Transformer for 3-D Human Pose Estimation},
  author={Shu Chen, Ying He},
  journal={IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1-11, 2025, Art no. 5031811, doi: 10.1109/TIM.2025.3569914.},
  year={2025}
}
```

