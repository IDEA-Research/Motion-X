# 🚀 How to prepare Non Mocap Data -- AIST?

<details>
<summary> Firstly, please make sure that you have downloaded and collected the dataset as the following directory structure: </summary>


```
../datasets  

├──  motion_data
  ├── smplx_322
    ├── idea400
    ├── ...
├──  face_motion_data
  ├── smplx_322
    ├── humanml
    ├── EgoBody
    ├── GRAB
├── texts
  ├──  semantic_labels
    ├── idea400
    ├── ...
  ├──  face_texts
    ├── humanml
    ├── EgoBody
    ├── GRAB
    ├── idea400
    ├── ...
```

</details>

- Run the following code to process AIST dataset:

```
python aist.py
```

