# NEES

The official PyTorch implementation of Neural Extraction framework for multiscale Essential Structures (NEES) in the following paper:

```
Q. Liu, B. Wang, Neural extraction of multiscale essential structure for network dismantling, Neural Networks 154 (2022) 99–108.
```
#Dependencies

- python
- torch
- torch-geometric
- networkx
- numpy

## Usage
1.  Input: the input network data should be written in a txt file. Each row contains 2 node IDs, which should be integers and started from 1. File "football.txt" in folder "./netdata/" is an example of a network with 115 nodes.
    
```
1 2
3 4
```
2.  Run: parameters can be modified in Gback-ASEDGE_e.py, and run the following to train and test the model:

```
python Gback-ASEDGE_e.py
```

3.  Output: the output is a list of nodes that should be removed. For the given example network data, the output file is "result-football.txt".



## Citation

Please cite our work if you find our code/paper is helpful to your work.

```
@article{LIU202299,
title = {Neural extraction of multiscale essential structure for network dismantling},
journal = {Neural Networks},
volume = {154},
pages = {99-108},
year = {2022},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2022.07.015},
url = {https://www.sciencedirect.com/science/article/pii/S0893608022002726},
author = {Qingxia Liu and Bang Wang},
}
```
