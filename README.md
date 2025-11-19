# SCMamba for CASSI

This repo is the implementation of paper "2D-Slice and 3D-Cube Mamba Network for Snapshot Spectral Compressive Imaging (TCSVT)"

# Abstract

Hyperspectral image (HSI) reconstruction algorithms are fundamental to coded aperture snapshot spectral imaging (CASSI) systems. Recently, deep unfolding networks (DUNs) have emerged as a dominant solution, seamlessly combining traditional optimization frameworks with the strengths of deep learning. Among these, Mamba stands out as a prominent method for modeling long-range dependencies. However, its reliance on one-dimensional (1D) spatial scanning often compromises spectral consistency and spatial coherence, leading to misalignment of neighboring pixels within sequences. To address these limitations, we propose a novel multi-view framework based on 2D-slice modeling, which ensures spatial-spectral continuity in 1D sequences while maintaining computational efficiency. Furthermore, motivated by the need for precise local patch modeling in 2D images, we develop a 3D-cube Mamba model for HSI reconstruction. By integrating the UNet architecture, this model enhances spatial and spectral detail representation through multi-scale receptive field modeling, using fixed cube sizes to dynamically adjust pixel distances. These advancements are incorporated into the A-HQS-accelerated deep unfolding framework, synergistically combining the strengths of 2D-slice and 3D-cube MambaNet to achieve state-of-the-art HSI reconstruction performance. Experimental evaluations on simulated and real-world CASSI datasets demonstrate the efficacy of the proposed approach, achieving superior spectral fidelity and detailed feature representation.


# Usage 

## Prepare Dataset:

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:


```
|--SCM-DUN
    |--datasets
        |--CSI
            |--cave_1024_28
                |--scene1.mat
                |--scene2.mat
                ：  
                |--scene205.mat
            |--CAVE_512_28
                |--scene1.mat
                |--scene2.mat
                ：  
                |--scene30.mat
            |--KAIST_CVPR2021  
                |--1.mat
                |--2.mat
                ： 
                |--30.mat
            |--TSA_simu_data  
                |--mask_3d_shift.mat
                |--mask.mat   
                |--Truth
                    |--scene01.mat
                    |--scene02.mat
                    ： 
                    |--scene10.mat
            |--TSA_real_data  
                |--mask_3d_shift.mat
                |--mask.mat   
                |--Measurements
                    |--scene1.mat
                    |--scene2.mat
                    ： 
                    |--scene5.mat
    |--checkpoints
    |--csi
    |--scripts
    |--tools
    |--results
    |--Quality_Metrics
    |--visualization
```

We use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (cave_1024_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set.

## Mamba Environment

```
conda create -n your_env_name python=3.10.13
conda activate your_env_name
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
git clone https://github.com/Dao-AILab/causal-conv1d.git 
cd causal-conv1d 
git checkout v1.2.0 # current latest version tag 
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd ./mamba
git checkout v1.2.0 # current latest version tag
MAMBA_FORCE_BUILD=TRUE pip install .
```


## Simulation Experiement:

### Training

```
cd SCM-DUN/

# DERNN-LNLT 3stage
bash ./scripts/train_SCMAMBA_3stg_simu.sh

...

# DERNN-LNLT 7stage
bash ./scripts/train_SCMAMBA_7stg_simu.sh

# DERNN-LNLT 9stage
bash ./scripts/train_SCMAMBA_9stg_simu.sh

```

The training log, trained model, and reconstrcuted HSI will be available in `SCM-DUN/exp/` .

### Testing

Place the pretrained model to `SCM-DUN/checkpoints/`

Run the following command to test the model on the simulation dataset.

```
cd SCM-DUN/

# SCMAMBA 3stage
bash ./scripts/test_SCMAMBA_3stg_simu.sh

...

# SCMAMBA 7stage
bash ./scripts/test_SCMAMBA_7stg_simu.sh

# SCMAMBA 9stage
bash ./scripts/test_SCMAMBA_9stg_simu.sh

```

The reconstrcuted HSIs will be output into `SCM-DUN/results/`

```
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.


### Visualization

- Put the reconstruted HSI in `SCM-DUN/visualization/simulation_results/results` and rename it as method.mat, e.g., SCMAMBA_9stg_simu.mat
- Generate the RGB images of the reconstructed HSIs

```
cd SCM-DUN/visualization/
Run show_simulation.m 
```


## Real Experiement:

### Training

```
cd SCM-DUN/

# DERNN-LNLT 3stage
bash ./scripts/train_SCMAMBA_3stg_real.sh
```

The training log and trained model will be available in `SCM-DUN/exp/`

### Testing

```
cd SCM-DUN/

# DERNN-LNLT 3stage
bash ./scripts/test_SCMAMBA_3stg_real.sh
```

The reconstrcuted HSI will be output into `SCM-DUN/results/`


## Acknowledgements

Our code is based on following codes, thanks for their generous open source:

- [https://github.com/ShawnDong98/RDLUF_MixS2](https://github.com/ShawnDong98/RDLUF_MixS2)
- [https://github.com/caiyuanhao1998/MST](https://github.com/caiyuanhao1998/MST)
- [https://github.com/TaoHuang95/DGSMP](https://github.com/TaoHuang95/DGSMP)
- [https://github.com/mengziyi64/TSA-Net](https://github.com/mengziyi64/TSA-Net)
- [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- [https://github.com/ShawnDong98/DERNN-LNLT](https://github.com/ShawnDong98/DERNN-LNLT)


## Citation

If you use our method in your work please cite our paper:
* BibTex：


    @ARTICLE{11230624,
      author={Feng, Yuchao and Qin, Mengjie and Wu, Zongliang and Yang, Yuxiang and Gao, Junhua and Yuan, Xin},
      journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
      title={2D-Slice and 3D-Cube Mamba Network for Snapshot Spectral Compressive Imaging}, 
      year={2025},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TCSVT.2025.3629725}
    }



* Plane Text：
	
    Y. Feng, M. Qin, Z. Wu, Y. Yang, J. Gao and X. Yuan, "2D-Slice and 3D-Cube Mamba Network for Snapshot Spectral Compressive Imaging," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2025.3629725.
