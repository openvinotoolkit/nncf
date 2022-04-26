### BootstrapNAS

NNCF supports the automated generation and training of weight-sharing super-networks for NAS. 

<p align="center">
<img src="architecture.png" alt="BootstrapNAS Architecture" width="500"/>
</p>

BootstrapNAS takes as input a pre-trained model that is used to generate a weight-sharing super-network. BootstrapNAS then applies a training strategy, and once the super-network has been trained, it searches for efficient subnetworks that satisfy the user's requirements. 

Please use this *bibtex* entry to cite this work: 
```BibTex
@article{DBLP:journals/corr/abs-2112-10878,
  author    = {J. Pablo Muñoz and Nikolay Lyalyushkin and Yash Akhauri and Anastasia Senina and
               Alexander Kozlov and Nilesh Jain},
  title     = {Enabling NAS with Automated Super-Network Generation},
  journal   = {CoRR},
  volume    = {abs/2112.10878},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10878},
  eprinttype = {arXiv},
  eprint    = {2112.10878},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10878.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```