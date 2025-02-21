from utils import *
import tensorflow as tf
from torch import tensor



# Main Script
if __name__ == "__main__":
    #with open('embeddings_EXPERIMENT_3.p', 'rb') as f:
    #       embeddings = pickle.load(f, encoding='latin1')

    # Load datasets
    dataset_names, datasets_list = load_datasets()
    mnist_data = limit_dataset_size(datasets_list[0], max_size=10000)
    
    datasets_names = ['MNIST']
    embeddings = []
    with tf.device('/GPU:0'):
        print("embeddings for: MNIST")
        probe_network = get_model('resnet18', pretrained=True, num_classes=10)
        embeddings.append(Task2Vec(probe_network, max_samples=None, skip_layers=6).embed(mnist_data))

    rotation_angles = [45, 90, 135, 180, 225, 270, 315]

    for angle in rotation_angles:
        rotated_data = []
        for item, label in mnist_data:
            item_rotated = F.rotate(item, angle, interpolation=F.InterpolationMode.BILINEAR)
            rotated_data.append((item_rotated, label))
        with tf.device('/GPU:0'):
            print("embeddings for:",f'rotatedMNIST{angle}')
            probe_network = get_model('resnet18', pretrained=True, num_classes=10)
            embeddings.append(Task2Vec(probe_network, max_samples=None, skip_layers=6).embed(rotated_data))
            datasets_names.append(f'rotatedMNIST{angle}')
  
    
    # Save embeddings
    with open('embeddings_EXPERIMENT.p', 'wb') as f:
        pickle.dump((datasets_names, embeddings), f)
    
total_complexity = calculate_total_complexity(embeddings)
sequential_heterogeneity = calculate_sequential_heterogeneity(embeddings)

print(f"Total Complexity: {total_complexity:.4f}")
print(f"Sequential Heterogeneity: {sequential_heterogeneity:.4f}")



