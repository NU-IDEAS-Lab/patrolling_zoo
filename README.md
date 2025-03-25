# Patrolling Zoo

This repository contains the policy/training code for the paper, ["Graph Neural Network-based Multi-agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems"](https://doi.org/10.1109/IROS58592.2024.10802510), by Anthony Goeckner, Yueyuan Sui, Nicolas Martinet, Xinliang Li, and Qi Zhu of Northwestern University in Evanston, Illinois.

If you use this code, please cite our paper as:
```
@INPROCEEDINGS{10802510,
  author={Goeckner, Anthony and Sui, Yueyuan and Martinet, Nicolas and Li, Xinliang and Zhu, Qi},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Graph Neural Network-based Multi-agent Reinforcement Learning for Resilient Distributed Coordination of Multi-Robot Systems}, 
  year={2024},
  volume={},
  number={},
  pages={5732-5739},
  doi={10.1109/IROS58592.2024.10802510}}
```


## Package Description
Packages are as follows:

 * **onpolicy**: Contains the algorithm code.
 * **patrolling_zoo**: Contains the environment code.

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
