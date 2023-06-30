# **Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset**
### [Project Page](https://motion-x-dataset.github.io/) | [Video](https://www.youtube.com/watch?v=s0cG3OVXQUo&t=2s) | [Paper](https://motion-x-dataset.github.io/) | [Data](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform)
This repository contains the implementation of the following paper:
> Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset <br>[Jing Lin](https://github.com/linjing7)<sup>∗12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Shunlin Lu](https://shunlinlu.github.io/)<sup>∗3</sup>, [Yuanhao Cai](https://github.com/caiyuanhao1998)<sup>2</sup>, [Ruimao Zhang](http://www.zhangruimao.site/)<sup>3</sup>, [Haoqian Wang](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution <sup>1</sup>International Digital Economy Academy <sup>2</sup> Tsinghua University <sup>3</sup>The Chinese University of Hong Kong, Shenzhen

<p align="middle">
<img src="assets/overview.gif" width="100%">
<br>
<em>Overview of the proposed Motion-X dataset</em>
</p>

**Table of Contents**

  1. [General Description](#general-description)
  2. [Dataset Download](#dataset-download)
  3. [Experiments](#experiments)
  5. [Citing Motion-X](#citing-human-art)

## General Description

We propose a systematic annotation pipeline and build a large-scale 3D expressive whole-body human motion dataset from massive online videos and eight existing motion sub-datasets. We unifies them into the same formats, providing whole-body motion (i.e., SMPL-X) and corresponding text labels.

**Contents Motion-X:** 

- Motion label: including `13.7M` whole-body poses and `96K` motion clips annotation, represented as SMPL-X parameters. 

- Text label: `13.7M` frame-level pose description and `96K` sequence-level semantic lables

- Other labels: RGB images, audio

- A high-accuracy and efficient  whole-body human motion-text annotation pipeline


**Tasks that Motion-X targets for:**

- Text-driven 3d whole-body human motion genetaion
  - Notably, model pretrained with Motion-X  raise 0.04 R-Precision and decline 0.58 FID  on HumanML3D testset, indicating that Motion-X may benefit the performance on existing dataset.

- 3D whoe-body human mesh recovery
- Motion pretraining, large motion model, etc.

<p align="middle">
<img src="assets/cross_the_single_plank_bridge_.gif" width="100%">
<br>
<em>Example of the RGB videos and annotated motion</em>
</p>

## Dataset Download  

* Although we re-annotate the lables of the sub-dataset with our annotation pipeline, to aviod license conflict, we only provide our annotated result to the users with the approvals from the original institutions. Please follow each link separately and request the given subset. 

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
    <tr>
      <th>EgoBody</th>
      <td>1.0K</td>
      <td>0.4M</td>
      <td><a href="https://sanweiliti.github.io/egobody/egobody.html" target="_blank">sanweiliti.github.io</a></td>
    </tr>
    <tr></tr>
    <tr>
      <th>BAUM</th>
      <td>1.4K</td>
      <td>0.2M</td>
      <td><a href="https://mimoza.marmara.edu.tr/~cigdem.erdem/BAUM1/" target="_blank">mimoza.marmara.edu.tr</a></td>
    </tr>
    <tr></tr>
    <tr>
      <th>Online Videos</th>
      <td>15K</td>
      <td>3.4M</td>
      <td>YouTube</td>
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
#### Accuracy of the motion annotation pipelines

<p align="middle">
<img src="assets/motion_annot_exp.png" width=70%">
<br>
</p>

#### Text-driven Whole-body Human Motion Generation

<p align="middle">
<img src="assets/motion_generation_exp.png" width=70%">
<br>
</p>

#### 3D Whole-Body Human Mesh Recovery

<p align="middle">
<img src="assets/mesh_recovery_exp.png" width=40%">
<br>
</p>

## Reference  

If you find this repository useful for your work, please consider citing it as follows:

```  
@article{lin2023one,
  title={One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer},
  author={Lin, Jing and Zeng, Ailing and Wang, Haoqian and Zhang, Lei and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
