# NIPS_2025_Submission_20931

## Setting it up

Follow the instruction to clone project and create environment

```bash
# clone project
git clone https://github.com/NIPS-2025-20931/NIPS_2025_Submission_20931.git
cd NIPS_2025_Submission_20931
# create data folder
mkdir data
# install dependencies
conda env create -f environment.yml
```

To test the GINO code, follow the installation guide from their Github [[GINO]](https://github.com/neuraloperator/neuraloperator.git)

## Get Started

1. Prepare the Data: To access the experimental datasets, please use the following link. The dataset should be placed in data folder.

| Dataset | Link |
| -------------------------------------------- | ------------------------------------------------------------ |
| BackwardStep | [[Google Cloud]](https://drive.google.com/file/d/1bgib7Xu6ClIB_DHbJBkzFuCVjAwu775L/view?usp=sharing) |
| CylinderArray | [[Google Cloud]](https://drive.google.com/file/d/1KEmZCXdt94hNrVC-Y-sYCdsWJNyGd-si/view?usp=sharing) |
| Shallow Water | [[Google Cloud]](https://drive.google.com/file/d/12LrTSEMpfmdhcYEMTVaCY5q4gWhhfSzY/view?usp=sharing) |
| Smoke Plume | [[Google Cloud]](https://drive.google.com/file/d/10llQG6s2rUCuVjkswIDGAppZm2wFkVnj/view?usp=sharing) |

2. Training the model: Run <modelname>_train.sh in src/dataset folder to train the model for the given dataset.

3. The example use the pre-trained weights here [[Google Cloud]](https://drive.google.com/file/d/1StXVWnJbxz4ylG5PVzlNIvjrT6UOstdG/view?usp=sharing). Place the file in src/smoke/FLUID/weights

## Project Structure

The directory structure of new project looks like this:

```
├── data                   <- Extract the data here
│   ├── backwardStep
│   ├── cylinderArray
│   ├── shallow_water
│   └── smoke
│
├── src                    <- Source code
│   ├── models                   <- Model scripts
│   ├── backwardStep
│       └──conf                        <-Hydra configs
│   ├── cylinderArray
│       └──conf                        <-Hydra configs
│   ├── shallow_water
│       └──conf                        <-Hydra configs
│   └── smoke
│       └──conf                        <-Hydra configs
│
├── environment.yml         <- File for installing conda environment
├── smoke_example           <- FLUID example for smoke data
└── README.md
```
