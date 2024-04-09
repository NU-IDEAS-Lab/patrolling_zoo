# SDZoo

<!-- This repository contains the policy/training code for the paper, "Graph Neural Network-based Multi-agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems", by Anthony Goeckner, Yueyuan Sui, Nicolas Martinet, Xinliang Li, and Qi Zhu of Northwestern University in Evanston, Illinois. -->

Gyaan Thesis work forked from [patrolling zoo](https://github.com/NU-IDEAS-Lab/patrolling_zoo)

## Package Description
Packages are as follows:

 * **onpolicy**: Contains the algorithm code.
 * **sdzoo**: Contains the environment code.

## Installation

 1) Clone the sdzoo repository:
    ```bash
    git clone --recurse git@github.com:NU-IDEAS-Lab/sdzoo.git
    ```

 2) Create a Conda environment with required packages:
    ```bash
    cd ./sdzoo
    conda env create -n sdzoo -f ./environment.yml
    conda activate sdzoo
    ```

 3) Install PyTorch to the new `sdzoo` conda environment using the [steps outlined on the PyTorch website](https://pytorch.org/get-started/locally/).

 4) Install the `onpolicy` and `sdzoo` packages:
    ```
    pip install -e .
    ```

## Operation

You may run the example in `onpolicy/scripts/train_patrolling_scripts/mappo.ipynb`.