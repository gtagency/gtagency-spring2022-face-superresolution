# Face Superresolution

**Contributors**
- Akash Chakka 
- Abhay Sheshadri 
- Vidushi Maheshwari 
- manny was here

## Getting Started

First, let's set up a conda environment. As of now, PyTorch and some other libraries don't work with Python 3.10,
so we need to configure the environment with Python 3.9.

```bash
conda create -n face-superresolution python=3.9
```

Now, let's activate the environment and install the dependencies
```bash
conda activate face-superresolution
conda install -c pytorch pytorch
conda install -c conda-forge face_recognition
conda install pillow
```

