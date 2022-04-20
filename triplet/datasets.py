import numpy as np
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Each of the non-empty clusters are sampled uniformly to avoid trivial solution.
    Args:
        labels (list): lists of images indexes.
        n_classes (int): number of clusters 
        n_samples (int): number of samples per cluster
    Returns batches of size n_classes * n_samples.
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_idx = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        self.label_to_indices = self.generate_indexes_epoch()
        for l in self.label_to_indices:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.label_to_indices}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for k in self.label_idx.keys():
            if len(self.label_idx[k]) != []:
                nmb_non_empty_clusters += 1
        
        size_per_pseudolabel = int(len(self.labels) / nmb_non_empty_clusters) + 1
        
        label_to_indices = {}
        for k in self.label_idx.keys():
            if self.label_idx[k]!=[]:
                label_to_indices[k] = np.random.choice(self.label_idx[k], 
                size_per_pseudolabel,
                replace=(len(self.label_idx[k]) <= size_per_pseudolabel))
        return label_to_indices



    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            #print(self.label_to_indices.keys())
            classes = np.random.choice(list(self.label_to_indices.keys()), len(self.label_to_indices.keys()), replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
