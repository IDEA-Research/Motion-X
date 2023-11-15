# 🚀 How to use?

This instruction is for creating the Motion Representation with the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) format. Upon the body part, we extend the body representation to body-hand representation. And from our ablation experiments, we get rid of the local rotation part of this representation, and only leave root and local positions. We call it **Tomato Representaion** for convenience.

## 1. Data Preparing


<details>
<summary>Download SMPL+H, SMPLX, DMPLs.</summary>

Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in AMASS project), DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL), and SMPL-X model from [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php). Then place all the models under `./body_model/`. The `./body_model/` folder tree should be:

```bash
./body_models
├── dmpls
│   ├── female
│   │   └── model.npz
│   ├── male
│   │   └── model.npz
│   └── neutral
│       └── model.npz
├── smplh
│   ├── female
│   │   └── model.npz
│   ├── info.txt
│   ├── male
│   │   └── model.npz
│   └── neutral
│       └── model.npz
├── smplx
│   ├── female
│   │   ├── model.npz
│   │   └── model.pkl
│   ├── male
│   │   ├── model.npz
│   │   └── model.pkl
│   └── neutral
│       ├── model.npz
└───────└── model.pkl
```

</details>


<details>
<summary>Download Motion-X datasets</summary>

Please follow the instruction of [Motion-X](https://github.com/IDEA-Research/Motion-X) to download the SMPLX data with the dimension of 322. Put the motion data in folder `./data/motion_data/smplx_322`.

</details>


## 2. Data Processing
(1) get joints positions
```
python raw_pose_processing.py
```
(2) get Representation
```
python motion_representation.py
```
(3) visualization for checking. The output dimension should be 623. 
```
python plot_3d_global.py
```
