import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.io

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label

def load_paviau_data(data_file_path, labels_file_path):
    # Load PaviaU data from the .mat file
    data = scipy.io.loadmat(data_file_path)
    
    # Load ground truth labels from the .mat file
    labels_data = scipy.io.loadmat(labels_file_path)
    
    # Extract the relevant information (adjust according to your data structure)
    # Assuming your data has keys 'data' for spectral data and 'labels' for labels
    spectral_data = data['data']
    labels = labels_data['labels']
    
    return spectral_data, labels

def getDataset(dataset, paviau_data_file_path=None, paviau_labels_file_path=None):
    if dataset == 'PaviaU' and paviau_data_file_path is not None and paviau_labels_file_path is not None:
        # Load PaviaU data and ground truth labels
        spectral_data, labels = load_paviau_data(paviau_data_file_path, paviau_labels_file_path)
        
        # Define your transformation for PaviaU data (adjust accordingly)
        transform_paviau = transforms.Compose([
            # Your transformations here
            transforms.ToTensor(),
        ])
        
        # Create a CustomDataset using the loaded PaviaU data
        dataset = CustomDataset(spectral_data, labels, transform=transform_paviau)
        
        # Set the number of classes and input channels accordingly
        num_classes = len(set(labels.flatten()))
        inputs = spectral_data.shape[-1]  # Assuming the last dimension is the number of spectral bands
        
        return dataset, num_classes, inputs

    # Add other dataset loading logic here...
    # Keep the existing dataset loading logic for other datasets

# Example usage for PaviaU
paviau_data_file_path = 'data/PaviaU.mat'
paviau_labels_file_path = 'data/PaviaU_gt.mat'
paviau_dataset, paviau_num_classes, paviau_inputs = getDataset('PaviaU', paviau_data_file_path, paviau_labels_file_path)
