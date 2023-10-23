# **Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset**

![](./assets/logo.jpg)

<p align="center">
  <a href='https://arxiv.org/abs/2307.00818'>
    <img src='https://img.shields.io/badge/Arxiv-2307.00818-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2307.00818pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://motion-x-dataset.github.io'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>
  <a href='https://youtu.be/0a0ZYJgzdWE'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  <a href='https://github.com/IDEA-Research/Motion-X'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href='LICENSE'>
    <img src='https://img.shields.io/badge/License-IDEA-blue.svg'>
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=IDEA-Research.Motion-X&left_color=gray&right_color=orange">
  </a>
</p>

This repository contains the implementation of the following paper:
> Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset <br>[Jing Lin](https://jinglin7.github.io/)<sup>😎12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>😎1</sup>, [Shunlin Lu](https://shunlinlu.github.io/)<sup>😎13</sup>, [Yuanhao Cai](https://github.com/caiyuanhao1998)<sup>2</sup>, [Ruimao Zhang](http://www.zhangruimao.site/)<sup>3</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>😎</sup>Equal contribution. <sup>
1</sup>International Digital Economy Academy <sup>2</sup> Tsinghua University <sup>3</sup>The Chinese University of Hong Kong, Shenzhen

## 🥳 Highlight Motion Samples

<img src="assets/overview.gif" width="100%">

**📊 Table of Contents**

  1. [General Description](#general-description)
  2. [Dataset Download](#dataset-download)
  3. [Experiments](#experiments)
  4. [Citing](#citing)

## 📜 General Description

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

## 📥 Dataset Download

We disseminate Motion-X in a manner that aligns with the original data sources. Here are the instructions:

### 1. Request Authorization

Please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) to request authorization to use Motion-X for non-commercial purposes.

### 2. Non-Mocap Subsets

For the non-mocap subsets, please read and acknowledge the licenses and terms of use on the original websites and then download the data from the provided Google Drive / Baidu Disk link.  Notably:

- We do not distribute the original RGB videos. We only provide the motion and text labels annotated by our team.
- Due to data license and quality consideration, we do not provide NTU-RGBD120 dataset. Instead, we build IDEA-400, which includes 400 daity actions (covering NTU-RGBD120). Please refer to this [video](https://www.youtube.com/watch?v=QWoll6asFhE) for a detailed introduction of IDEA-400. 

### 3. Mocap Subsets  

For the mocap datasets (i.e., AMASS, GRAB, EgoBody), please refer to [this link](https://github.com/IDEA-Research/Motion-X/tree/main/mocap-dataset-process) for a detailed instruction, notably:

- We do not distribute the original motion data. 
- We only provide the text labels and facial expressions annotated by our team. 

<div align="center">
<table cellspacing="0" cellpadding="0" bgcolor="#ffffff" border="0">
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Clip Number</th>
    <th align="center">Frame Number</th>
    <th align="center">Website</th>
    <th align="center">License</th>
    <th align="center">Downloading Link</th>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>AMASS</b></td>
    <td align="center">26K</td>
    <td align="center">3.5M</td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/" target="_blank">AMASS<br>Website</a></td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/license.html" target="_blank">AMASS<br>License</a></td>
    <td align="center"><a href="https://amass.is.tue.mpg.de/login.php" target="_blank">AMASS Data</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>EgoBody</b></td>
    <td align="center">1.0K</td>
    <td align="center">0.4M</td>
    <td align="center"><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">EgoBody<br>Website</a></td>
    <td align="center"><a href="https://egobody.ethz.ch/" target="_blank">EgoBody<br>License</a></td>
    <td align="center"><a href="https://egobody.ethz.ch/" target="_blank">EgoBody Data</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>GRAB</b></td>
    <td align="center">1.3K</td>
    <td align="center">1.6M</td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/" target="_blank">GRAB<br>Website</a></td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/license.html" target="_blank">GRAB<br>License</a></td>
    <td align="center"><a href="https://grab.is.tue.mpg.de/login.php" target="_blank">GRAB Data</a></td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>IDEA-400</b></td>
    <td align="center">120K</td>
    <td align="center">1.3M</td>
    <td align="center"><a href="https://motion-x-dataset.github.io/" target="_blank">IDEA400<br>Website</a>
    <td align="center"><a href="https://docs.google.com/document/d/1xeNQkkxD39Yi6pAtJrFS1UcZ2LyJ6RBwxicwQ2j3-Vs" target="_blank">IDEA400 License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">IDEA400 Data</a>
  </td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>AIST++</b></td>
    <td align="center">1.4K</td>
    <td align="center">1.1M</td>
    <td align="center"><a href="https://google.github.io/aistplusplus_dataset/" target="_blank">AIST++ <br>Website</a></td>
    <td align="center"><a href="https://google.github.io/aistplusplus_dataset/factsfigures.html" target="_blank">AIST++<br>License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">AIST++ Data</a>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>HAA500</b></td>
    <td align="center">9.9K</td>
    <td align="center">0.6M</td>
    <td align="center"><a href="https://www.cse.ust.hk/haa/" target="_blank">HAA500<br>Website</a></td>
    <td align="center"><a href="https://www.cse.ust.hk/haa/index.html" target="_blank">HAA500<br>License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">HAA500 Data</a>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>HuMMan</b></td>
    <td align="center">0.9K</td>
    <td align="center">0.2M</td>
    <td align="center"><a href="https://caizhongang.github.io/projects/HuMMan/" target="_blank">HuMMan<br>Website</a></td>
    <td align="center"><a href="https://caizhongang.github.io/projects/HuMMan/license.txt" target="_blank">HuMMan<br>License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">HuMMan Data</a>
     </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>BAUM</b></td>
    <td align="center">1.4K</td>
    <td align="center">0.2M</td>
    <td align="center"><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">BAUM<br>Website</a>
    <td align="center"><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">BAUM<br>License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">BAUM Data</a>
</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>Online Videos</b></td>
    <td align="center">15K</td>
    <td align="center">3.4M</td>
    <td align="center">---</td>
    <td align="center">---</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">Online Data</a>
  </tr>
  <tr></tr>
  <tr></tr>
  <tr style="background-color: lightgray;">
    <td align="center"><b>Motion-X (Ours)</b></td>
    <td align="center">96K</td>
    <td align="center">13.7M</td>
    <td align="center"><a href="https://motion-x-dataset.github.io/" target="_blank">Motion-X Website</a></td>
    <td align="center"><a href="https://docs.google.com/document/d/1xeNQkkxD39Yi6pAtJrFS1UcZ2LyJ6RBwxicwQ2j3-Vs" target="_blank">Motion-X License</a></td>
    <td align="center"><a href="https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform" target="_blank">Motion-X Data</a>
  </tr>
</table>
</div>

- Finally, the `dataset` is collected as the following directory structure:

```  
${ROOT}  
|-- dataset  
|   |-- motion_data
|   |   |-- HumanML3D
|   |   |   |-- 000001.npy
|   |   |-- EgoBody
|   |   |   |-- 000001.npy
|   |   |-- GRAB
|   |   |   |-- 000001.npy
|   |   |-- IDEA_400
|   |   |   |-- 000001.npy
|   |   |-- ......
|   |-- text_data
|   |   |-- semantic_labels
|   |   |   |-- HumanML3D
|   |   |   |   |-- 000001.txt
|   |   |   |-- EgoBody
|   |   |   |   |-- 000001.txt
|   |   |   |-- GRAB
|   |   |   |   |-- 000001.txt
|   |   |   |-- IDEA_400
|   |   |   |   |-- 000001.txt
|   |   |   |-- ......
|   |   |-- pose_descriptions
|   |   |   |-- HumanML3D
|   |   |   |   |-- 000001.txt
|   |   |   |-- EgoBody
|   |   |   |   |-- 000001.txt
|   |   |   |-- GRAB
|   |   |   |   |-- 000001.txt
|   |   |   |-- IDEA_400
|   |   |   |   |-- 000001.txt
|   |   |   |-- ......
```

</details>

### 🚀  Data Loading 


* To load the motion and text labels you can simply do:

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

## 💻 Experiments  
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

## 🤝 Citation  

If you find this repository useful for your work, please consider citing it as follows:

```  
@article{lin2023motionx,
  title={Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset},
  author={Lin, Jing and Zeng, Ailing and Lu, Shunlin and Cai, Yuanhao and Zhang, Ruimao and Wang, Haoqian and Zhang, Lei},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
