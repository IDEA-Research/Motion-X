# **Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset**
### [Project Page](https://motion-x-dataset.github.io/) | [Paper](https://arxiv.org/pdf/2307.00818.pdf) | [Data](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) (coming soon!)
This repository contains the implementation of the following paper:
> Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset <br>[Jing Lin](https://jinglin7.github.io/)<sup>∗12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Shunlin Lu](https://shunlinlu.github.io/)<sup>∗3</sup>, [Yuanhao Cai](https://github.com/caiyuanhao1998)<sup>2</sup>, [Ruimao Zhang](http://www.zhangruimao.site/)<sup>3</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
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
<img src="assets/perform____images____cross_the_single_plank_bridge____cross_the_single_plank_bridge_subset_1.gif" width="100%">
</p>

<p align="middle">
<img src="assets/kungfu____images____subset_0____Aerial_Kick_Kungfu_wushu_clip12____Aerial_Kick_Kungfu_wushu_clip12.gif" width="100%">
<br>
</p>
<p align="middle">
<img src="assets/animation_actions____images____subset_0____Horse_clip1.gif" width="100%">
<br>
<em>Figure 2. Example of the RGB video and annotated motion, RGB videos are from: <a href="https://www.xiaohongshu.com/user/profile/5ec2aac700000000010059c0/618e6c7f000000000102e60b">website1</a></em>, 
  <a href="https://www.patreon.com/mastersongkungfu">website2</a></em>, 
    <a href="https://www.youtube.com/channel/UCzgkpehSWuFTQx9E8NkBqzw">website3</a></em>
</p>

## Dataset Download

We hope to disseminate Motion-X in a manner that aligns with the original data sources and complies with the necessary protocols. Here are the instructions:

- Fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) to request authorization to use Motion-X for non-commercial purposes. After you submit the form, an email containing the dataset will be delivered to you as soon as we release the dataset. We plan to release Motion-X by Sept. 2023.

- For the motion capture datasets (i.e., AMASS, GRAB, EgoBody),
  - we will not distribute the original motion data. So Please download the originals from the original websites.
  - We will provide the text labels and facial expressions annotated by our team. 
- For the other datasets (i.e., NTU-RGBD120, AIST++, HAA500, HuMMan),
  - please read and acknowledge the licenses and terms of use on the original websites.
  - Once users have obtained necessary approvals from the original institutions, we will provide the motion and text labels annotated by our team.

<div align="center">
<table cellspacing="0" cellpadding="0" bgcolor="#ffffff" border="0">
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Clip <br> Number</th>
    <th align="center">Frame <br> Number</th>
    <th align="center"> Body <br> Motion </td>
  	<th align="center"> Hand <br> Motion </td>
  	<th align="center"> Facial <br> Motion </td>
		<th align="center"> Semantic <br> Text </td>
  	<th align="center"> Pose <br> Text </td>
    <th align="center">Website</th>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>AMASS</b></td>
    <td align="center">26K</td>
    <td align="center">3.5M</td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/" target="_blank">AMASS</a></td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/" target="_blank">AMASS</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://github.com/EricGuo5513/HumanML3D" target="_blank">HumanML3D</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/" target="_blank">amass</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>NTU-RGBD120</b></td>
    <td align="center">38K</td>
    <td align="center">2.6M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center"><a href="https://rose1.ntu.edu.sg/dataset/actionRecognition/" target="_blank">NTU</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://rose1.ntu.edu.sg/dataset/actionRecognition/" target="_blank">rose1</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>AIST++</b></td>
    <td align="center">1.4K</td>
    <td align="center">1.1M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center"><a href="https://google.github.io/aistplusplus_dataset/" target="_blank">AIST++</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://google.github.io/aistplusplus_dataset/" target="_blank">aist</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>HAA500</b></td>
    <td align="center">9.9K</td>
    <td align="center">0.6M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center"><a href="https://www.cse.ust.hk/haa/" target="_blank">HAA500</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://www.cse.ust.hk/haa/" target="_blank">cse.ust.hk</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>HuMMan</b></td>
    <td align="center">0.9K</td>
    <td align="center">0.2M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://caizhongang.github.io/projects/HuMMan/" target="_blank">HuMMan</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>GRAB</b></td>
    <td align="center">1.3K</td>
    <td align="center">1.6M</td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/" target="_blank">GRAB</a></td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/" target="_blank">GRAB</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/" target="_blank">GRAB</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/" target="_blank">grab</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>EgoBody</b></td>
    <td align="center">1.0K</td>
    <td align="center">0.4M</td>
    <td align="center"><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">EgoBody</a></td>
    <td align="center"><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">EgoBody</a></td>
    <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">sanweiliti</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>BAUM</b></td>
    <td align="center">1.4K</td>
    <td align="center">0.2M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center"><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">BAUM</a></td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">mimoza</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>Online Videos</b></td>
    <td align="center">15K</td>
    <td align="center">3.4M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center">online</td>
  </tr>
  <tr></tr>
  <tr></tr>
  <tr style="background-color: lightgray;">
    <td align="center"><b>Motion-X (Ours)</b></td>
    <td align="center">96K</td>
    <td align="center">13.7M</td>
    <td align="center">Ours</td> <td align="center">Ours</td> <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center">Ours</td>
    <td align="center"><a href="https://motion-x-dataset.github.io/" target="_blank">motion-x</a></td>
  </tr>
</table>
</div>

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
  body_text = np.loadtxt('texts/body_texts/000001.txt')     # body pose description
  hand_text = np.loadtxt('texts/hand_texts/000001.txt')     # hand pose description
  face_text = np.loadtxt('texts/face_texts/000001.txt')     # facial expression
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
  journal={arXiv preprint arXiv: 2307.00818},
  year={2023}
}
```
