


def get_balanced_subset(dataset, m , lables = None):
    # Create a dictionary to store m graphs per class for the training set
    label_dict = {}
    # train_data = []
    train_indices = list()

    if lables is None:
        for data in dataset:
            label = data.y.item()
            if label not in label_dict:
                label_dict[label] = []
    else:
        for lable in lables:
            label_dict[label] = []
    
    # Create the balanced training set
    for idx, data in enumerate(dataset):
        label = data.y.item()
        
        if label not in label_dict: continue
            
        if len(label_dict[label]) < m:
            label_dict[label].append(data)
            # train_data.append(data)
            train_indices.append(idx)
        
        # Stop if we have m samples for each label
        if all(len(samples) >= m for samples in label_dict.values()):
            break
    return train_indices, label_dict


def run_metric_testing(dataset):
    dataset.shuffle()
    indices, label_dict = get_balanced_subset(dataset,2)

