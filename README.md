# Coat based identification of Holstein-Friesian cattle using deep metric learning
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asheeshkrsharma/Identification-OpenCows/blob/main/evaluationV0.8.ipynb)

---
## Project description
Identifying individuals in a group of animals is an active research topic as it paves the way to behavioural studies for lameness and illness assessment. Real-world applications demand algorithms to identify previously unseen individuals. For instance, traditional classification algorithms such as Convolutional Neural Networks, which must be trained exclusively for every cow at a farm, cannot classify or differentiate newly added cattle without prior re-training. Metric learning is a solution where deep learning models learn a latent space of differentiating features (or embedding) as a distance metric for simple clustering algorithms to associate similar individuals.

In this project, we independently replicated the deep metric learning technique proposed by Andrew et al. (2020) to train a Residual Convolutional Neural Network (Resnet) on a dataset of Holstein-Friesian cattle, which can identify unknown individuals, previously unseen during training. The dataset we used for training the Resnet is called [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17) which was released by the authors under [Non-Commercial Government Licence for public sector information](http://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/non-commercial-government-licence.htm). The technique relies on learning a 128-dimensional latent feature space that, when used to fit simple algorithms such as K-Nearest Neighbours, can perform open-set identification by clustering identical cows. We also implemented the proposed hybrid loss function and compared its performance with traditional losses often used in identification tasks. Furthermore, we also assessed the quality of the 128-dimensional embedding trained on the proposed loss function vs the traditional losses by t-SNE dimensionality reduction.

See [Reports.pdf](https://github.com/Asheeshkrsharma/Identification-OpenCows/blob/main/Report.pdf) for more details.

<details>
<summary>Directory structure</summary>

```
  Identification-OpenCows/
    ├── models/ (Contains pretrained model weights)
    │    ├── *.pkl... (Training history)
    │    └── *.pth... (corresponding model weights)
    ├── staticAssets/ (contains supporting figures and files)
  Identification-OpenCows/
    ├── utils/ (Contain utility scripts)
  Report.pdf (Detailed description of the project)
  evaluationV0.8.ipynb (For running the project)
```
</details>


# Getting started

## Quickstart
Best way to run this project is through google collab using evaluationV0.8.ipynb jupyter notebook. The notebook takes care of all the dependencies and downloads the dataset.

[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asheeshkrsharma/Identification-OpenCows/blob/main/evaluationV0.8.ipynb)

## On a local machine (assumes python3 and jupyter installed)
1. Start by cloning this repository

    `git clone https://github.com/Asheeshkrsharma/Identification-OpenCows`

    `cd Identification-OpenCows`

2. Install dependencies by running

    `pip3 install -r requirements.txt`

3. Run jupyter notebook in the root directory

    `jupyter notebook .`
4. Open evaluationV0.8.ipynb from jupyter.

# References
Andrew, W., Gao, J., Mullan, S., Campbell, N., Dowsey, A. W., & Burghardt, T. (2021). Visual identification of individual Holstein-Friesian cattle via deep metric learning. Computers and Electronics in Agriculture, 185, 106133.

William Andrew, Tilo Burghardt, Neill Campbell, Jing Gao (2020): OpenCows2020. https://doi.org/10.5523/bris.10m32xl88x2b61zlkkgz3fml17

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Wang, Zhuoyi & Tao, Hemeng & Kong, Zelun & Chandra, Swarup & Khan, Latifur. (2019). Metric Learning based Framework for Streaming Classification with Concept Evolution. 1-8. 10.1109/IJCNN.2019.8851934. 