# Continual Learning on Structurally Related Tasks

Here, we conduct an empirical study of the continual learning setting when the sequence of learning tasks is structurally related.The study investigates how task structure affects the performance of continual learning algorithms.

## Cloning the repository
Make sure to add ```--recurse-submodules``` to pull the submodules' code.
```
git clone --recurse-submodules https://github.com/evry-paris-saclay/2425-m2cns-masteriales-CLwithStructures
```
## Project Summary

Continual learning enables models to learn tasks sequentially without forgetting previously acquired knowledge, a challenge known as **catastrophic forgetting**.

This project explores the relationship between **task sequence properties** and learning performance, using two key metrics:

* **Total Complexity**: Overall difficulty of the task sequence.
* **Sequential Heterogeneity**: Dissimilarity between consecutive tasks.

We leverage the **Task2Vec** framework to compute task embeddings and evaluate these metrics' influence on performance.

## Main Contributions

* **Correlation Analysis**:

  * Strong positive correlation between total complexity and forgetting.
  * Weak or negative correlation between sequential heterogeneity and forgetting (in some cases, it improves performance via knowledge transfer).

* **Algorithm Evaluation**: Four continual learning algorithms were tested on rotated MNIST:

  * `Synaptic Intelligence (SI)`
  * `Average Gradient Episodic Memory (A-GEM)`
  * `Elastic Weight Consolidation (EWC)`
  * `Learning Without Forgetting (LwF)`

## Experimental Setup

* **Dataset**: Rotated MNIST (8 or 15 tasks with 24° or 45° rotation steps).
* **Evaluation Metrics**: Accuracy after each task, correlation with complexity and heterogeneity.
* **Task Similarity**: Computed using Task2Vec and cosine distance.

## Folder Structure

```
├── data/                  # MNIST data and rotations
├── docs/                  # Detailed report of the experiments
├── models/                # Continual learning algorithms Framework & Task2Vec Framwork
├── plots/                 # The plots that represent learning of each experiment
├── main.py                # Task2Vec embeddings and correlation analysis
├── utils & utilsPlots     # contains the logic executed in the main.py
└── README.md              # Project summary
```
* **Key Findings**:

  * A-GEM was the most robust to forgetting.
  * Mixed task orders helped reduce forgetting more than sequential ones.
  * Increasing the number of tasks improved generalization.
  * EWC was less sensitive to task diversity, while LwF struggled with complex tasks.

<p align="center">
  <img src="plots/A-GEM%20errVSsqh.png" alt="A-GEM Average Error vs Sequential Heterogegnity" width="400"/>
  <img src="plots/A-GEM%20errVStc.png" alt="A-GEM Average Error vs Total Complexity" width="400"/>
</p>

All the plots showing the average learning accuracy are [here](models/ContinualLearningFrameWork/store/plots)

## Authors

- **Yara El Halawani**  
  [20235309@etud.univ-evry.fr](mailto:20235309@etud.univ-evry.fr)  
  [LinkedIn](https://www.linkedin.com/in/yaraelhalawani/)

- **Talout Chattah**  
  [20235542@etud.univ-evry.fr](mailto:20235542@etud.univ-evry.fr)
  [taloutchattah2001@gmail.com](mailto:taloutchattah2001@gmail.com)  
  [LinkedIn](https://www.linkedin.com/in/talout-chattah/) 

- **Mohamed Demes**  
  [20234747@etud.univ-evry.fr](mailto:20234747@etud.univ-evry.fr)  
  [LinkedIn](https://www.linkedin.com/in/mohamed-demes-516480258/) 

- **Taher Lmouden**  
  [20235157@etud.univ-evry.fr](mailto:20235157@etud.univ-evry.fr)  

**Supervised by**: Massinissa Hamidi [massinissa.hamidi@univ-evry.fr ](mailto:massinissa.hamidi@univ-evry.fr)


## References

* Task2Vec: [https://arxiv.org/abs/1811.05827](https://arxiv.org/abs/1811.05827)
* A-GEM: [https://arxiv.org/abs/1812.00420](https://arxiv.org/abs/1812.00420)
* EWC: [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)
* LwF: [https://arxiv.org/abs/1606.09282](https://arxiv.org/abs/1606.09282)
