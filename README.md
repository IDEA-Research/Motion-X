# **Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset**
### [Project Page](https://motion-x-dataset.github.io/) | [Paper](https://arxiv.org/pdf/2307.00818.pdf) | [Data](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) (coming soon!)
This repository contains the implementation of the following paper:
> Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset <br>[Jing Lin](https://github.com/linjing7)<sup>∗12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Shunlin Lu](https://shunlinlu.github.io/)<sup>∗3</sup>, [Yuanhao Cai](https://github.com/caiyuanhao1998)<sup>2</sup>, [Ruimao Zhang](http://www.zhangruimao.site/)<sup>3</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution. <sup>
1</sup>International Digital Economy Academy <sup>2</sup> Tsinghua University <sup>3</sup>The Chinese University of Hong Kong, Shenzhen

<p align="middle">
<img src="assets/overview.gif" width="100%">
<br>
<em>Figure 1. Motion samples from our dataset</em>
</p>


**Table of Contents**

  1. [General Description](#general-description)
  2. [Dataset Download](#dataset-download)
  3. [Experiments](#experiments)
  5. [Citing](#citing)

## General Description

We propose a high-accuracy and efficient annotation pipeline for whole-body motions and the corresponding text labels. Based on it, we build a large-scale 3D expressive whole-body human motion dataset from massive online videos and eight existing motion datasets. We unify them into the same formats, providing whole-body motion (i.e., SMPL-X) and corresponding text labels.

**Labels from Motion-X:** 

- Motion label: including `13.7M` whole-body poses and `96K` motion clips annotation, represented as SMPL-X parameters. 
- Text label: (1) `13.7M` frame-level whole-body pose description and (2) `96K` sequence-level semantic labels.
- Other modalities: RGB videos, audio, and music information.

**Supported Tasks:**

- Text-driven 3d whole-body human motion generation
- 3D whole-body human mesh recovery
- Others: Motion pretraining, multi-modality pre-trained models for motion understanding and generation, etc.

<p align="middle">
<img src="assets/cross_the_single_plank_bridge_.gif" width="100%">
<br>
<em>Figure 2. Example of the RGB video and annotated motion, RGB video is from this <a href="https://www.xiaohongshu.com/user/profile/5ec2aac700000000010059c0/618e6c7f000000000102e60b">website</a></em>
</p>


## Dataset Download

* Although we re-annotate the labels of the sub-dataset with our annotation pipeline, to avoid license conflict, we only provide our annotated results to the users with the approvals from the original institutions. Please follow each link separately, request, and cite the given datasets. 

  <div align="center">
  <table cellspacing="0" cellpadding="0" bgcolor="#ffffff" border="0">
    <tr>
      <th>Dataset</th>
      <th>Clip Number</th>
      <th>Frame Number</th>
      <th>Source</th>
    </tr>
    <tr></tr>
    <tr>
      <td><b>AMASS</b></td>
      <td>26K</td>
      <td>3.5M</td>
      <td><a href="https://amass.is.tue.mpg.de/" target="_blank">amass.is.tue.mpg.de</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>NTU-RGBD120</b></td>
      <td>38K</td>
      <td>2.6M</td>
      <td><a href="https://rose1.ntu.edu.sg/dataset/actionRecognition/" target="_blank">rose1.ntu.edu.sg</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>AIST++</b></td>
      <td>1.4K</td>
      <td>1.1M</td>
      <td><a href="https://google.github.io/aistplusplus_dataset/" target="_blank">aistplusplus_dataset</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>HAA500</b></td>
      <td>9.9K</td>
      <td>0.6M</td>
      <td><a href="https://www.cse.ust.hk/haa/" target="_blank">www.cse.ust.hk</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>HuMMan</b></td>
      <td>0.9K</td>
      <td>0.2M</td>
      <td><a href="https://caizhongang.github.io/projects/HuMMan/" target="_blank">HuMMan</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>GRAB</b></td>
      <td>1.3K</td>
      <td>1.6M</td>
      <td><a href="https://grab.is.tue.mpg.de/" target="_blank">grab.is.tue.mpg.de</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>EgoBody</b></td>
      <td>1.0K</td>
      <td>0.4M</td>
      <td><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">sanweiliti.github.io</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>BAUM</b></td>
      <td>1.4K</td>
      <td>0.2M</td>
      <td><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">mimoza.marmara.edu.tr</a></td>
    </tr>
    <tr></tr>
    <tr>
      <td><b>Online Videos</b></td>
      <td>15K</td>
      <td>3.4M</td>
      <td>YouTube</td>
    </tr>
    <tr></tr>
    <tr></tr>
    <tr style="background-color: lightgray;">
      <td><b>Motion-X (Ours)</b></td>
      <td>96K</td>
      <td>13.7M</td>
      <td><a href="https://motion-x-dataset.github.io/" target="_blank">motion-x-dataset</a></td>
    </tr>
  </table>
  </div>

* Collect the confirmations of the eight sub-datasets in a PDF to demonstrate that you have acquired the approvals from the original institutions.

* Fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) to request authorization to use Motion-X for non-commercial purposes.  After you submit the form, an email containing the dataset will be delivered to you as soon as we release the dataset. We plan to release Motion-X by Sept. 2023.

* To retrieve motion and text labels you can simply do:
  ```python
  import numpy as np
  import torch
  
  # read motion and save as smplx representation
  motion = np.load('motion_data/000001.npy')
  motion = torch.tensor(motion).float()
  motion_parms = {
              'root_orient': motion[:, :3],  # controls the global root orientation
              'pose_body': motion[:, 3:3+63],  # controls the body
              'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
              'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
              'face_expr': motion[:, 159:159+50],  # controls the face expression
              'face_shape': motion[:, 209:209+100],  # controls the face shape
              'trans': motion[:, 309:309+3],  # controls the global body position
              'betas': motion[:, 312:],  # controls the body shape. Body shape is static
          }
  
  # read text labels
  semantic_text = np.loadtxt('texts/semantic_texts/000001.npy')     # semantic labels 
  body_text = np.loadtxt('texts/body_texts/000001.npy')     # body pose description
  hand_text = np.loadtxt('texts/hand_texts/000001.npy')     # hand pose description
  face_text = np.loadtxt('texts/face_texts/000001.npy')     # facial expression
  ```

## Experiments  
#### Validation of the motion annotation pipeline

Our annotation pipeline significantly surpasses existing SOTA 2D whole-body models and mesh recovery methods.
<p align="middle">
<img src="assets/motion_annot_exp.png" width=80%">
<br>
</p>

#### Benchmarking Text-driven Whole-body Human Motion Generation

<p align="middle">
<img src="assets/motion_generation_exp.png" width=80%">
<br>
</p>

#### Comparison with HumanML3D on Whole-body Human Motion Generation Task

<p align="middle">
<img src="assets/humanml_comp_exp.png" width=80%">
<br>
</p>

#### Impact on 3D Whole-Body Human Mesh Recovery

<p align="middle">
<img src="assets/mesh_recovery_exp.png" width=50%">
<br>
</p>

## Citing  

If you find this repository useful for your work, please consider citing it as follows:

```  
@article{lin2023motionx,
  title={Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset},
  author={Lin, Jing and Zeng, Ailing and Lu, Shunlin and Cai, Yuanhao and Zhang, Ruimao and Wang, Haoqian and Zhang, Lei},
  journal={arXiv preprint arXiv: },
  year={2023}
}
```
