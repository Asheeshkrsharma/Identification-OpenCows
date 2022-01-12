# Coat based identification of Holstein-Friesian cattle using deep metric learning
[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Asheeshkrsharma/Identification-OpenCows/blob/main/evaluationV7.ipynb)

---
## Project description
Identifying individuals in a group of animals is an active research topic as it paves the way to behavioural studies for lameness and illness assessment. Real-world applications demand algorithms to identify previously unseen individuals. For instance, traditional classification algorithms such as Convolutional Neural Networks, which must be trained exclusively for every cow at a farm, cannot classify or differentiate newly added cattle without prior re-training. Metric learning is a solution where deep learning models learn a latent space of differentiating features (or embedding) as a distance metric for simple clustering algorithms to associate similar individuals.

In this project, we independently replicated the deep metric learning technique proposed by Andrew et al. (2020) to train a Residual Convolutional Neural Network (Resnet) on a dataset of Holstein-Friesian cattle, which can identify unknown individuals, previously unseen during training. The dataset we used for training the Resnet is called [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17) which was released by the authors under [Non-Commercial Government Licence for public sector information](http://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/non-commercial-government-licence.htm). The technique relies on learning a 128-dimensional latent feature space that, when used to fit simple algorithms such as K-Nearest Neighbours, can perform open-set identification by clustering identical cows. We also implemented the proposed hybrid loss function and compared its performance with traditional losses often used in identification tasks. Furthermore, we also assessed the quality of the 128-dimensional embedding trained on the proposed loss function vs the traditional losses by t-SNE dimensionality reduction.

See [Reports.pdf]() for more details.

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
```
</details>

## 

