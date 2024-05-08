import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn import preprocessing
import numpy as np

# Define a custom Dataset class
class SequenceKdDataset(Dataset):
    def __init__(self, sequences, features, labels, encoding_type):
        self.sequences = sequences
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.encoding_type = encoding_type 
        self.amino_acid_mapping =  {
                                    'A': 0,  # Alanine
                                    'C': 1,  # Cysteine
                                    'D': 2,  # Aspartic Acid
                                    'E': 3,  # Glutamic Acid
                                    'F': 4,  # Phenylalanine
                                    'G': 5,  # Glycine
                                    'H': 6,  # Histidine
                                    'I': 7,  # Isoleucine
                                    'K': 8,  # Lysine
                                    'L': 9,  # Leucine
                                    'M': 10, # Methionine
                                    'N': 11, # Asparagine
                                    'P': 12, # Proline
                                    'Q': 13, # Glutamine
                                    'R': 14, # Arginine
                                    'S': 15, # Serine
                                    'T': 16, # Threonine
                                    'V': 17, # Valine
                                    'W': 18, # Tryptophan
                                    'Y': 19  # Tyrosine
                                }
    
    # Function to perform one-hot encoding
    def one_hot_encode(self, sequence):
        # Initialize the encoded sequence array
        encoded = np.zeros((len(sequence), len(self.amino_acid_mapping)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in self.amino_acid_mapping:
                encoded[i, self.amino_acid_mapping[aa]] = 1.0
        return encoded
    
    #Function for indices and embedding layers support
    def sequence_to_indices(self, sequence):
        # Convert the sequence to a list of indices
        indices = [self.amino_acid_mapping[aa] for aa in sequence if aa in self.amino_acid_mapping]
        return indices
    
    def __len__(self):
        return len(self.labels)
     

    def __getitem__(self, idx):
       
        sequence = self.sequences[idx]
        features = self.features[idx]
        label = self.labels[idx]

        if self.encoding_type == "one_hot":
            sequence_encoded = self.one_hot_encode(sequence)
        elif self.encoding_type == "embeddings":
            sequence_encoded = self.sequence_to_indices(sequence)
        
        #Use proper torch type for encoded sequences
        sequence_encoded = torch.tensor(sequence_encoded, dtype=torch.long if self.encoding_type == "one_hot" else torch.long)

        return sequence_encoded, features, label


#Define the early stopping supporting function
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Function to calculate loss on validation dataset
def validate(val_model, val_loader, loss_measurement):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_model.eval()
    all_batch_val_loss = 0
    batch_val_samples = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for pos, (seqs, features, labels) in enumerate(val_loader):

            seqs = seqs.to(device)
            features = features.to(device)
            labels = labels.to(device)



            outputs = val_model(x = seqs,
                            additional_features = features
                            )

            loss = loss_measurement(outputs, labels)
            all_batch_val_loss += loss.item() * seqs.size(0)
            batch_val_samples += seqs.size(0)

            all_val_preds.append(outputs.detach())
            all_val_labels.append(labels)

    average_validation_loss = all_batch_val_loss  / batch_val_samples
    return average_validation_loss, all_val_preds, all_val_labels

#Dataloaders extraction per kfold
def get_data_loaders_per_kfold(full_dataset, fold_idx, batch_size, kfold_structure):
    kfold = kfold_structure  # Ensure reproducibility

    # Generate indices for splits:
    splits = list(kfold.split(range(len(full_dataset))))  # We split indices of the dataset

    # Select a split
    train_idx, val_idx = splits[fold_idx]

    # Create Subset objects using indices
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)


    # Wrap subsets in DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



#Function to help parameterize the CNN models we are building. Helpful for modularity
def cnn_model_instantiation(cnn_kd_prediction = None, **kwargs):

    #Architecture details
    sequence_length = kwargs.get('encoded_sequence_length',246) #dynamically determined
    num_additional_features = kwargs.get('num_of_additional_features', 20)#dynamically determined
    use_aa_features = kwargs.get('use_aa_features', True)
    
    use_embedding = kwargs.get('use_embedding', True)
    use_dropout = kwargs.get('use_dropout', True)
    conv_dropout = kwargs.get('conv_dropout', 0.2)

    #Conv details
    
    embedding_dimension = kwargs.get('embedding_dimension', 64)
    conv_layer_kernel_size = kwargs.get('conv_layer_kernel_size', 3)
    pooling_kernel_size = kwargs.get('pooling_kernel_size', 2)
    pool_stride = kwargs.get('pool_stride', 2)
    conv_stride = kwargs.get('conv_stride', 1)
    conv_padding = kwargs.get('conv_padding' , 1)
    conv_dilation = kwargs.get('conv_dilation',1)


    #Optimizer
    lr_rate = kwargs.get('lr_rate',0.001)
    L2_reg = kwargs.get('L2_reg',1e-5)

    #Model name
    model_name = kwargs.get('model_name', 'Protein_Kd_Generic_Model')

    model = cnn_kd_prediction(use_cnn = True,
                        sequence_length = sequence_length,
                        use_aa_features = use_aa_features,
                        num_additional_features = num_additional_features,
                        embedding_dim = embedding_dimension,
                        conv_layer_kernel_size = conv_layer_kernel_size,
                        pooling_kernel_size = pooling_kernel_size,
                        pool_stride = pool_stride,
                        conv_dilation = conv_dilation,
                        conv_stride = conv_stride,
                        use_embedding = use_embedding,
                        conv_padding = conv_padding,
                        model_name = model_name,
                        use_dropout = use_dropout, 
                        dropout_rate = conv_dropout)

    loss_measurement = nn.MSELoss()

    if L2_reg:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate,  weight_decay = L2_reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, model_name, optimizer, loss_measurement, device

def post_process_epoch_metrics(metrics):
    averaged_across_epochs = {}
    avg_score_per_metric = {}
    avg_score_per_fold = {}
    
    for data_slice, metrics_and_folds in metrics.items():
       
        avg_score_per_metric.setdefault(data_slice, {})
        avg_score_per_fold.setdefault(data_slice, {})
        for metric, fold_report in metrics_and_folds.items():
            
            avg_score_per_fold[data_slice][metric] = {}
            for fold_name, fold_values in fold_report.items():
                
                if fold_values:
                    avg_score_per_fold[data_slice][metric].setdefault(fold_name, 0)
                    avg_score_per_fold[data_slice][metric][fold_name] = np.mean(fold_values)
                    avg_score_per_metric[data_slice].setdefault(metric, []).extend(fold_values)

            #averaged_across_epochs.setdefault(metric, {})[data_slice] = {fold: np.mean(scores) for fold, scores in enumerate(folds)}

        avg_score_per_metric[data_slice] = {metric: np.mean(scores) for metric, scores in avg_score_per_metric[data_slice].items()}

    return avg_score_per_fold, avg_score_per_metric


