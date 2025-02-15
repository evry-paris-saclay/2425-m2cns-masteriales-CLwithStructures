import itertools
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, cifar10
from models.Task2vec.task2vec import Task2Vec
from models.Task2vec.models import get_model
import models.Task2vec.datasets as datasets
import models.Task2vec.task_similarity
from itertools import combinations, permutations
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import gzip
import string
import math
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

# Step 1: Load MNIST
def load_datasets():
    dataset_names = ['mnist']
    dataset_list = [datasets.__dict__[name](root='./data')[0] for name in dataset_names]
    return dataset_names, dataset_list

# Limit Dataset Size for Debugging (My PC contraints)
def limit_dataset_size(dataset, max_size=1000):
    """
    Limits the dataset size by selecting a random subset of the data.

    Args:
        dataset: The dataset to limit.
        max_size: The maximum number of samples to retain in the dataset.

    Returns:
        A subset of the original dataset with at most `max_size` samples.
    """
    # Randomly select a subset of the dataset up to `max_size` samples
    if len(dataset) > max_size:
        indices = torch.randperm(len(dataset)).tolist()[:max_size]
        limited_data = torch.utils.data.Subset(dataset, indices)
    else:
        limited_data = dataset

    return limited_data

def generateTaskEmbeddings(names, datasets ):
    embeddings = []
    for name, dataset in  zip(names,datasets):
        print(f"Embedding {name}")
        probe_network = get_model('resnet18', pretrained=True, num_classes=10)
        embeddings.append(Task2Vec(probe_network, max_samples=5000, skip_layers=6).embed(dataset))
    return embeddings

def get_vector(embedding):
    """Flatten hessian and scale into a single vector representation."""
    return np.concatenate([embedding.hessian.flatten()])

def cosine_distance(e_t, e_t_prime):
    """Compute cosine distance between e_t and e_t' with zero-vector handling."""
    e_t, e_t_prime = np.array(e_t), np.array(e_t_prime)
    
    # Avoid division by zero in normalization
    if np.all(e_t == 0) or np.all(e_t_prime == 0):
        return 1  # Maximum dissimilarity if one vector is zero
    
    v_t = e_t / (e_t + e_t_prime)
    v_t_prime = e_t_prime / (e_t + e_t_prime)

    norm_v_t = np.linalg.norm(v_t)
    norm_v_t_prime = np.linalg.norm(v_t_prime)
    
    # Avoid division by zero in cosine similarity
    if norm_v_t == 0 or norm_v_t_prime == 0:
        return 1  # Maximum distance

    cosine_sim = np.dot(v_t, v_t_prime) / (norm_v_t * norm_v_t_prime)
    return 1 - cosine_sim  # Cosine distance

def calculate_total_complexity(embeddings):
    """
    Compute total complexity C(T) as the sum of cosine distances to a reference vector e_0.
    """
    e_0 = get_vector(embeddings[0])  # The trivial Task !! what is the trivial task ? is it the embedding vector of normal MNIST ?
    total_complexity = sum(cosine_distance(get_vector(e), e_0) for e in embeddings)
    return total_complexity

def calculate_sequential_heterogeneity(embeddings):
    """
    Compute sequential heterogeneity F(T) as the sum of cosine distances between consecutive embeddings.
    """
    seq_heterogeneity = sum(
        cosine_distance(get_vector(embeddings[i]), get_vector(embeddings[i+1])) 
        for i in range(len(embeddings) - 1)
    )
    return seq_heterogeneity
