EDPN: Enhanced Deep Pyramid Network for Blurry Image Restoration
====
Ruikang Xu, Zeyu Xiao, Jie Huang, Yueyi Zhang, Zhiwei Xiong. [EDPN: Enhanced Deep Pyramid Network for Blurry Image Restoration](https://arxiv.org/abs/2105.04872). In CVPRW 2021. (Winner of NTIRE Challenge 2021 on Image Deblurring Track 1. Low Resolution) <br/>

[arXiv](https://arxiv.org/abs/2105.04872) | [CVF](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Xu_EDPN_Enhanced_Deep_Pyramid_Network_for_Blurry_Image_Restoration_CVPRW_2021_paper.pdf) 

## Dependencies
- This repository is based on [[EDVR/old_version]](https://github.com/xinntao/EDVR/tree/old_version), you can install DeformConv by following [[EDVR/old_version]](https://github.com/xinntao/EDVR/tree/old_version)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.2.0](https://pytorch.org/): `conda install pytorch=1.2.0 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `pip install numpy`
- opencv: `pip install opencv-python`
- tensorboardX: `pip install tensorboardX`

## Prepare your data
Take blurry image super-resolution as an example <br/>
```
cd deblur_SR/datasets && python createTxt.py
```


## Train the model
Take blurry image super-resolution as an example <br/>
```
cd deblur_SR && python train.py
```

## Test the model
Take blurry image super-resolution as an example <br/>
```
cd deblur_SR && python inference.py
```

## Dataset
[CodaLab](https://competitions.codalab.org/competitions/28073) | [NTIRE2021](https://data.vision.ee.ethz.ch/cvl/ntire21/)

## Citation
```
@InProceedings{Xu_2021_CVPRW_EDPN,
    author = {Xu, Ruikang and Xiao, Zeyu and Huang, Jie and Zhang, Yueyi and Xiong, Zhiwei},
    title = {EDPN: Enhanced Deep Pyramid Network for Blurry Image Restoration},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition WorkShops (CVPRW)},
    month = {June},
    year = {2021}
}  

@InProceedings{Nah_2021_CVPRW_Deblur,
    author = {Nah, Seungjun and Son, Sanghyun and Lee, Suyoung and Timofte, Radu and Lee, Kyoung Mu},
    title = {NTIRE 2021 Challenge on Image Deblurring},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition WorkShops (CVPRW)},
    month = {June},
    year = {2021}
}
```

## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn and zeyuxiao@mail.ustc.edu.cn.