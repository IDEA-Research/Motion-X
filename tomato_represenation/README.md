# 🍅 How to use tomato representation?

This instruction is for creating the Motion Representation with the [Tomato](https://arxiv.org/pdf/2310.12978.pdf) format. The tomato format is extended from the [H3D](https://github.com/EricGuo5513/HumanML3D) format and is different from it. We name it `Tomato Representation` for convenience. For detailed ablation on motion representation design choice, please refer to Appendix B.1 in the [paper](https://arxiv.org/pdf/2310.12978.pdf). 

## 🚀  Data Preparation


<details>
<summary>Download SMPL+H, SMPLX, DMPLs.</summary>

Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in the AMASS project), DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL), and SMPL-X model from [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php). Then, please place all the models under `./body_model/`. The `./body_model/` folder tree should be:

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

Please follow the instruction of [Motion-X](https://github.com/IDEA-Research/Motion-X) to download the SMPL-X data with the dimension of 322. Put the motion data in folder `./data/motion_data/smplx_322`.

</details>


## 🔧 Data Processing
(1) get joints positions
```
python raw_pose_processing.py
```
(2) get Representation
```
python motion_representation.py
```
(3) visualization for checking. The output dimension of motions (one frame) should be 623. 
```
python plot_3d_global.py
```

# 🤝🏼 Citation
If you use the tomato format, please consider to cite us as: 
```bash
@article{humantomato,
  title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
  author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
  journal={arxiv:2310.12978},
  year={2023}
}
```