# Patrolling Zoo

## Installation

 1) Clone the patrolling_zoo repository:
    ```bash
    git clone --recurse git@github.com:NU-IDEAS-Lab/patrolling_zoo.git
    ```

 2) Create a Conda environment with required packages:
    ```bash
    cd ./patrolling_zoo
    conda env create -n patrolling_zoo -f ./environment.yml
    conda activate patrolling_zoo
    ```

 3) Install PyTorch to the new `patrolling_zoo` conda environment using the [steps outlined on the PyTorch website](https://pytorch.org/get-started/locally/).

 4) Install the `onpolicy` and `patrolling_zoo` packages:
    ```
    pip install -e .
    ```

## Operation

You may run the example in `onpolicy/scripts/train_patrolling_scripts/mappo.ipynb`.