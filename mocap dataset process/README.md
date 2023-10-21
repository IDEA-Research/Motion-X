

# 🛠️ Installation

```bash
pip install smplx
```

# 🚀 How to prepare Mocap Data --Humanml3D, EgoBody, Mocap?

## 1. Data Preparing


<details>
<summary>Download SMPLX.</summary>

Download SMPL-X model from [SMPL-X_v1.1](https://smpl-x.is.tue.mpg.de/download.php). Then place all the models under `./body_model/`. The `./body_model/` folder tree should be:

```bash
./body_models

├── smplx
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_MALE.pkl
│   ├── SMPLX_NEUTRAL.npz
│   └── SMPLX_NEUTRAL.pkl
```

</details>


<details>
<summary>Download AMASS motions for Humanml3D.</summary>
    
  - Download [AMASS](https://amass.is.tue.mpg.de/download.php) motions. 
  - If you are using the SMPL-X (in Motion-X), please download the AMASS data with `SMPL-X G`. If you use the SMPL-X data, please save them at `./datasets/amass_data/`. 
  
  The `datasets/amass_data/` folder tree should be:
  
```bash
  ./amass_data/

  ├── amass_data
    ├── ACCAD
    ├── BioMotionLab_NTroje
    ├── BMLhandball
    ├── BMLmovi
    ├── CMU
    ├── CNRS
    ├── DFaust_67
    ├── EKUT
    ├── Eyes_Japan_Dataset
    ├── GRAB
    ├── HUMAN4D
    ├── HumanEva
    ├── KIT
    ├── MPI_HDM05
    ├── MPI_Limits
    ├── MPI_mosh
    ├── SFU
    ├── SOMA
    ├── SSM_synced
    ├── TCD_handMocap
    ├── TotalCapture
    └── Transitions_mocap
```
</details>    

<details>
<summary>Download Egobody motions.</summary>
    
  - Download [Egobody](https://egobody.ethz.ch/) motions. 
  - Please obey the Egobody dataset lisence and fill the form to get the download link.
  
  The `datasets/EgoBody/` folder tree should be:
```bash
  ./EgoBody/

  ├── EgoBody
    ├── smplx_camera_wearer_test
    ├── smplx_camera_wearer_train
    ├── smplx_camera_wearer_val
    ├── smplx_interactee_test
    ├── smplx_interactee_train
    └── smplx_interactee_val
```

</details>    

<details>
<summary>Download GRAB motions.</summary>
    
  - Download [GRAB](https://grab.is.tue.mpg.de/download.php) motions. 
  - Please obey the GRAB dataset lisence.
  
  The `datasets/EgoBody/` folder tree should be:

  ```bash
  ./GRAB/

  ├── GRAB
    ├── s1
    ├── s2
    ├── s3
    ├── s4
    ├── s5
    ├── s6
    ├── s7
    ├── s8
    ├── s9
    └── s10
  ```
</details>   



    

## 2. Generate mapping files and text files

In this step, we will process the Mocap datasets.

<details>
<summary>Process HumanML3D Dataset</summary>
    
Download `texts.zip` file. [HumanML3D](https://github.com/EricGuo5513/HumanML3D) repo.

```bash
python humanml.py
```
</details>    


<details>
<summary>Process EgoBody Dataset</summary>
    
The manual text annotation of Egobody dataset is provided at `egobody_description_all.csv`.

```bash
python egobody.py
```
</details>    
    
<details>
<summary>Process GRAB Dataset</summary>
    
```bash
python grab.py
```
</details> 


# 🤝🏼 Citation

If you use this three Mocap datasets for research, you need to cite them seperately: 

Humanml3D: 
```bash
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```

GRAB dataset:
```bash
@inproceedings{GRAB:2020,
  title = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  author = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020},
  url = {https://grab.is.tue.mpg.de}
}

@InProceedings{Brahmbhatt_2019_CVPR,
  title = {{ContactDB}: Analyzing and Predicting Grasp Contact via Thermal Imaging},
  author = {Brahmbhatt, Samarth and Ham, Cusuh and Kemp, Charles C. and Hays, James},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019},
  url = {https://contactdb.cc.gatech.edu}
}
```

EgoBody dataset:
```bash
@inproceedings{Zhang:ECCV:2022,
   title = {EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices},
   author = {Zhang, Siwei and Ma, Qianli and Zhang, Yan and Qian, Zhiyin and Kwon, Taein and Pollefeys, Marc and Bogo, Federica and Tang, Siyu},
   booktitle = {European conference on computer vision (ECCV)},
   month = oct,
   year = {2022}
}
```


If you have any question, please contact at Shunlin Lu (shunlinlu0803 [AT] gamil [DOT] com).
