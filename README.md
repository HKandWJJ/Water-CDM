# Water-CDM: Adaptive Double-Branch Fusion Conditional Diffusion Model for Underwater Image Restoration
This is the repository for our paper. **[Link](https://ieeexplore.ieee.org/document/11005520)**
![](Water-CDM.png)
# Data Preparation
Put your training and test dataset in ```dataset```. The structure of the ```dataset``` are as follows:
```
dataset
  |--train
      |--target
      |--input
  |--val
      |--target
      |--input
```
# Train
Run ```train.py``` to start training.
# Test
Set the checkpoint path in ```config/underwater.json```.
Run ```test.py```
# Citation
If you use the code in this repository for your work, please cite our paper:
```
@ARTICLE{11005520,
  author={Wang, Yingbo and He, Kun and Qu, Qiang and Du, Xiaogang and Liu, Tongfei and Lei, Tao and Nandi, Asoke K.},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Adaptive Double-Branch Fusion Conditional Diffusion Model for Underwater Image Restoration}, 
  year={2025},
  volume={},
  number={},
  pages={1-1}
```
# Acknowledgement
Our code architecture is based on the **[SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)**  ,**[DiffWater](https://github.com/Guan-MS/DiffWater?tab=readme-ov-file)**. We also thank for the following repositories:**[UIEB](https://li-chongyi.github.io/proj_benchmark.html)**,**[EUVP](https://irvlab.cs.umn.edu/resources/euvp-dataset)**,**[LSUI](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)**,**[Fusion](https://github.com/fergaletto/Color-Balance-and-fusion-for-underwater-image-enhancement.-.)**,**[MMLE](https://github.com/Li-Chongyi/MMLE_code)**,**[FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN)**,**[MLFcGAN](https://github.com/Xiaodong-Bran/MLFcGAN)**,**[Shallow-UWnet](https://github.com/mkartik/Shallow-UWnet)**,**[PUIE-net](https://github.com/zhenqifu/PUIE-Net)**.
