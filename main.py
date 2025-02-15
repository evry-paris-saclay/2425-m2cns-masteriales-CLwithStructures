from utils import *
import tensorflow as tf
from torch import tensor



# Main Script
if __name__ == "__main__":
    with open('embeddings.p', 'rb') as f:
           embeddings = pickle.load(f, encoding='latin1')
    
    # Load datasets
    """dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=3000)
    
    rotated_datasets = []
    datasets_names = ['MNIST']

    rotation_angles = [45, 90, 135, 180, 225, 270, 315]

    for angle in rotation_angles:
        rotated_data = []
        for item, label in mnist_data:
            item_rotated = F.rotate(item, angle, interpolation=F.InterpolationMode.BILINEAR)
            rotated_data.append((item_rotated, label))
        
        rotated_datasets.append(rotated_data)
        datasets_names.append(f'rotatedMNIST{angle}')

    data = (mnist_data, *rotated_datasets)

    with tf.device('/GPU:0'):
        embeddings = generateTaskEmbeddings(datasets_names,data)
  
    
    # Save embeddings
    with open('embeddings.p', 'wb') as f:
        pickle.dump((datasets_names, embeddings), f)"""
print(len(embeddings[1][0].hessian))  
total_complexity = calculate_total_complexity(embeddings[1])
sequential_heterogeneity = calculate_sequential_heterogeneity(embeddings[1])

print(f"Total Complexity: {total_complexity:.4f}")
print(f"Sequential Heterogeneity: {sequential_heterogeneity:.4f}")