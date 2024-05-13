import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, make_scorer, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import preprocessing
import numpy as np
import time
import os
import pickle




# Define a custom Dataset class
class SequenceKdDataset(Dataset):
    def __init__(self, sequences, features, labels, encoding_type, use_padding = False, max_pad_length = 0, padding_token_index = 20):
        self.sequences = sequences
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.encoding_type = encoding_type
        self.max_pad_length = max_pad_length
        self.use_padding = use_padding
        self.padding_token_index = padding_token_index
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
    def one_hot_encode(self, sequence):
        if self.use_padding:
            sequence = sequence[:self.max_pad_length]  # Truncate if longer than max_pad_length
            padded_length = self.max_pad_length
        else:
            padded_length = len(sequence)
        
        encoded = np.zeros((padded_length, len(self.amino_acid_mapping)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in self.amino_acid_mapping:
                encoded[i, self.amino_acid_mapping[aa]] = 1.0
        return encoded

    def sequence_to_indices(self, sequence):
        indices = [self.amino_acid_mapping.get(aa, self.padding_token_index) for aa in sequence[:self.max_pad_length]]  # Use padding index for unknown
        if self.use_padding:
            indices += [self.padding_token_index] * (self.max_pad_length - len(indices))  # Pad the indices list to max length
        return indices

    # # Function to perform one-hot encoding
    # def one_hot_encode(self, sequence):
    #     # Initialize the encoded sequence array
    #     encoded = np.zeros((len(sequence), len(self.amino_acid_mapping)), dtype=np.float32)
    #     for i, aa in enumerate(sequence):
    #         if aa in self.amino_acid_mapping:
    #             encoded[i, self.amino_acid_mapping[aa]] = 1.0
    #     return encoded
    
    # #Function for indices and embedding layers support
    # def sequence_to_indices(self, sequence):
    #     # Convert the sequence to a list of indices
    #     indices = [self.amino_acid_mapping[aa] for aa in sequence if aa in self.amino_acid_mapping]
    #     return indices
    
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
def cnn_model_instantiation(config_dict = {}):

    if not config_dict:
        raise ValueError("Missing model configuation details")

    #Architecture retrieval: 
    prediction_architecture = config_dict.get('prediction_architecture', None)

    if not prediction_architecture:
        raise ValueError("Missing model architecture for model isntantiation")

    #Architecture details
    sequence_length = config_dict.get('encoded_sequence_length',246) #dynamically determined
    num_additional_features = config_dict.get('num_of_additional_features', 20)#dynamically determined
    use_aa_features = config_dict.get('use_aa_features', True)
    
    use_embedding = config_dict.get('use_embedding', True)
    use_dropout = config_dict.get('use_dropout', True)
    conv_dropout = config_dict.get('conv_dropout', 0.2)
    use_batch_norm = config_dict.get('use_batch_norm', False)

    #Conv details
    
    embedding_dimension = config_dict.get('embedding_dimension', 64)
    conv_layer_kernel_size = config_dict.get('conv_layer_kernel_size', 3)
    pooling_kernel_size = config_dict.get('pooling_kernel_size', 2)
    pool_stride = config_dict.get('pool_stride', 2)
    conv_stride = config_dict.get('conv_stride', 1)
    conv_padding = config_dict.get('conv_padding' , 1)
    conv_dilation = config_dict.get('conv_dilation',1)

    padding_idx = config_dict.get('padding_idx', 20)


    #Optimizer
    lr_rate = config_dict.get('lr_rate',0.001)
    L2_reg = config_dict.get('L2_reg',1e-5)

    #Model name
    model_name = config_dict.get('model_name', 'Protein_Kd_Generic_Model')

    model = prediction_architecture(use_cnn = True,
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
                        dropout_rate = conv_dropout, 
                        use_batch_norm = use_batch_norm,
                        padding_idx = padding_idx)

    loss_measurement = nn.MSELoss()

    if L2_reg:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate,  weight_decay = L2_reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, model_name, optimizer, loss_measurement, device


#Function to help parameterize the CNN models we are building. Helpful for modularity
def rnn_model_instantiation(config_dict = {}):

    if not config_dict:
        raise ValueError("Missing model configuation details")

    #Architecture retrieval: 
    prediction_architecture = config_dict.get('prediction_architecture', None)

    if not prediction_architecture:
        raise ValueError("Missing model architecture for model isntantiation")

 
    use_embedding = config_dict.get('use_embedding', True)
    num_channels = config_dict.get('num_channels', 64)
    embedding_dimension = config_dict.get('embedding_dimension', 64)
    rnn_hidden_dim = config_dict.get('rnn_hidden_dim', 256)
    rnn_layers = config_dict.get('rnn_layers',2)
    use_dropout = config_dict.get('use_dropout', True)
    dropout_rate = config_dict.get('dropout_rate', 0.25)
    use_aa_features = config_dict.get('use_aa_features',True)
    num_additional_features = config_dict.get('num_additional_features', 0)
    bidirectional = config_dict.get('bidirectional', False)

    rnn_type = config_dict.get('rnn_type', "simple_rnn")


    #Optimizer
    lr_rate = config_dict.get('lr_rate',0.001)
    L2_reg = config_dict.get('L2_reg',1e-5)

    #Model name
    model_name = config_dict.get('model_name', 'Protein_Kd_Generic_Model')

    print("Config Dict:", config_dict)

    model = prediction_architecture(use_embedding = use_embedding,
                                    embedding_dimension = embedding_dimension, 
                                    num_channels = num_channels,
                                    use_aa_features = use_aa_features, 
                                    num_additional_features = num_additional_features,
                                    rnn_hidden_dim = rnn_hidden_dim, 
                                    rnn_layers = rnn_layers, 
                                    use_dropout = use_dropout, 
                                    dropout_rate = dropout_rate,
                                    bidirectional = bidirectional,
                                    rnn_type = rnn_type
                                    )

    loss_measurement = nn.MSELoss()

    if L2_reg:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate,  weight_decay = L2_reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, model_name, optimizer, loss_measurement, device


def training_loop(**kwargs):

    
    data_slices = kwargs.get('data_slices', [])
    num_epochs = kwargs.get('num_epochs', 10)
    device = kwargs.get('device', None)
   
    #Pass the model config dictionary necessary for model instantiation
    model_config_dict = kwargs.get('model_config_dict', {})

    #Cross_validation 
    n_folds = kwargs.get('n_folds', 5)
    batch_size = kwargs.get('batch_size', 64)

    #early stopping
    patience = kwargs.get('patience', 10)

    #Random_Seed
    random_seed = kwargs.get('random_seed', 15)

    #Kfold definition
    kfold_structure = kwargs.get('kfold_structure', KFold(n_splits=n_folds, shuffle=True, random_state=random_seed))

    architecture_to_use = kwargs.get('architecture_to_use', 'use_cnn')


    if not data_slices:
        raise ValueError('No training data slices provided')

    kfold_epoch_metrics ={}

    for pos, items in enumerate(data_slices.items()):
        k = items[0]
        v = items[1]



        print('-------- Dataset: ', k, '--------')
       
        history = {
                    'train_loss': [],
                    'val_loss': [],
                    'train_r2':[],
                    'val_r2':[],
                    'train_mae':[],
                    'val_mae':[],
                    'train_rmse':[],
                    'val_rmse':[]
                }

        kfold_epoch_metrics[k] = {}

        #Define epoch metrics per kfold per training_slice
        for metric in ['train_loss', 'val_loss', 'train_r2', 'val_r2', 'train_mae', 'val_mae', 'train_rmse', 'val_rmse']:
            kfold_epoch_metrics[k][metric] = {}
            for fold in range(n_folds):
                kfold_epoch_metrics[k][metric][str(fold) + '_fold'] = []

      
        print(kfold_epoch_metrics, ' ---- these are the epoch metrics for storage')

        print('using cross val')

        dataset= v.get('full_dataset', None)
        
        if not dataset:
            raise ValueError('No full dataset provided in loop')

        for fold_idx in range(n_folds):
            print(fold_idx, ' ---- this is the fold index')

            #Early stopping definition
            early_stopping = early_stopping = EarlyStopping(patience=patience, verbose=True, path='model_checkpoint.pt')
            
            train_loader, val_loader = get_data_loaders_per_kfold(full_dataset = dataset, fold_idx = fold_idx,batch_size = batch_size, kfold_structure = kfold_structure)
            print('We are at Kfold index: ', fold_idx)

            #RE-instantiate the model

            if architecture_to_use == 'use_cnn':
                model, model_name, optimizer, loss_measurement, device = cnn_model_instantiation(model_config_dict)
            elif architecture_to_use == 'use_rnn':
                model, model_name, optimizer, loss_measurement, device = rnn_model_instantiation(model_config_dict)
            elif architecture_to_use == 'use_stacked_cnn_rnn':
                model, model_name, optimizer, loss_measurement, device = cnn_model_instantiation(model_config_dict)

            #RE-instantiate the model
            print(f'{model_name}_DataSlice(' + k + ')_')
            
            print ('___________________')
            print('Model_Architecture: ')
            print(model )

            for epoch in range(num_epochs+1):
                start_time = time.time()
                
                if epoch %2 == 0:
                    print('We are at epoch: ' + str(epoch))

                model.train()
                all_batch_train_loss = 0
                batch_samples = 0

                all_train_preds = []
                all_train_labels = []
                all_val_preds = []
                all_val_labels = []

            
                for pos, batch in enumerate(train_loader):
                    

                    if pos == 0 and epoch == 0:  # Just print the first batch to inspect
                        if len(batch) == 3:
                            print("Batch structure is correct: (seqs, features, labels)")
                        else:
                            print("Batch structure incorrect:", batch)
                        

                    seqs = batch[0]
                    features = batch[1]
                    labels = batch[2]


                    seqs = seqs.to(device)
                    features = features.to(device)
                    labels = labels.to(device)

                    #print("Data type of sequences:", seqs.dtype)
                    #print(features, ' ---- these are the features')
                    #print("Data type of features:", features.dtype)

                    # Forward pass
                    outputs = model(x = seqs,
                                    additional_features = features)  # Adjust dimensions for Conv1D
                    
                    all_train_preds.append(outputs.detach())
                    all_train_labels.append(labels)

                    # Backward pass and optimization
                    loss = loss_measurement(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                    #Add loss per batch and normalize per batch size
                    all_batch_train_loss += loss.item() * seqs.size(0)
                    batch_samples += seqs.size(0)

                avg_batch_training_loss = all_batch_train_loss / batch_samples
                
                
                # Validate after each epoch
                avg_validation_loss, epoch_val_preds, epoch_val_labels = validate(val_model = model,
                                                                            val_loader = val_loader,
                                                                            loss_measurement = loss_measurement )
                
                early_stopping(avg_validation_loss, model)

                #Implement early stopping
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

                # Converting lists to tensors
                train_preds = torch.cat(all_train_preds)
                train_labels = torch.cat(all_train_labels)
                val_preds = torch.cat(epoch_val_preds)
                val_labels = torch.cat(epoch_val_labels)

                # Calculate metrics for training
                # Compute final metrics for epoch
                train_r2 = r2_score(train_labels.numpy(), train_preds.numpy())
                val_r2 = r2_score(val_labels.numpy(), val_preds.numpy())
                train_mae = mean_absolute_error(train_labels.numpy(), train_preds.numpy())
                val_mae = mean_absolute_error(val_labels.numpy(), val_preds.numpy())
                train_rmse = root_mean_squared_error(train_labels.numpy(), train_preds.numpy())
                val_rmse = root_mean_squared_error(val_labels.numpy(), val_preds.numpy())


                # Store metrics per epoch across folds
                for metric, value in zip(['train_loss', 'val_loss', 'train_r2', 'val_r2', 'train_mae', 'val_mae', 'train_rmse', 'val_rmse'],
                                        [avg_batch_training_loss, avg_validation_loss, train_r2, val_r2, train_mae, val_mae, train_rmse, val_rmse]):
                    kfold_epoch_metrics[k][metric][str(fold_idx) + '_fold'].append(value)
                end_time = time.time()



                #Printing time records of the epochs
                if epoch %2 == 0:
                    print(end_time - start_time, '- time to  finish one epoch')


            #Store the models
            try:
                # Construct the file path for saving the model results
                models_saving_path = f'./models/{model_name}/{model_name}_model_Training_Slice_{k}_Fold({fold_idx}).pt'
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(models_saving_path), exist_ok=True)
                
                #Save the model state dict
                torch.save(model.state_dict(), models_saving_path)
                
            except Exception as e:
                print(e, ' ---- error saving the model')
                pass

    #Store the results
    try:
        # Construct the file path for saving the model results
        model_results_path = f'./model_results/{model_name}_model_Training_Slice_{k}.pkl'
        
        # Create the directory if it doesn't exist
        #os.makedirs(os.path.dirname(f'./model_results/{model_name}'), exist_ok=True)
        
        # Open the file in write mode and dump the dictionary into it as JSON
        
        with open(model_results_path, 'wb') as file:
            pickle.dump(kfold_epoch_metrics, file)
       
    except Exception as e:
        print(e, ' ---- rror saving the results')
        pass

    return kfold_epoch_metrics

def post_process_epoch_metrics(metrics):
    averaged_across_epochs = {}
    avg_score_per_metric = {}
    avg_score_per_fold = {}
    max_score_per_metric = {}
    max_score_per_fold = {}

    
    for data_slice, metrics_and_folds in metrics.items():
       
        avg_score_per_metric.setdefault(data_slice, {})
        avg_score_per_fold.setdefault(data_slice, {})
        max_score_per_fold.setdefault(data_slice, {})
        for metric, fold_report in metrics_and_folds.items():
            
            avg_score_per_fold[data_slice][metric] = {}
            max_score_per_fold[data_slice][metric] = {}
            for fold_name, fold_values in fold_report.items():
                
                if fold_values:
                    avg_score_per_fold[data_slice][metric].setdefault(fold_name, 0)
                    avg_score_per_fold[data_slice][metric][fold_name] = np.mean(fold_values)
                    max_score_per_fold[data_slice][metric][fold_name] = np.max(fold_values)
                    avg_score_per_metric[data_slice].setdefault(metric, []).extend(fold_values)

            #averaged_across_epochs.setdefault(metric, {})[data_slice] = {fold: np.mean(scores) for fold, scores in enumerate(folds)}

        avg_score_per_metric[data_slice] = {metric: np.mean(scores) for metric, scores in avg_score_per_metric[data_slice].items()}

        max_score_per_metric[data_slice] = {metric: np.max(scores) for metric, scores in avg_score_per_metric[data_slice].items()}

        

    return avg_score_per_fold, avg_score_per_metric, max_score_per_metric, max_score_per_fold


def preparing_data_classic_ML(data):
    # Assuming `data` has two keys: 'sequence_features' and 'frequency_features'
    sequence_features = data['sequence_features'].numpy()  # converting sequence feature tensors to numpy array
    frequency_features = data['frequency_features'].numpy()  # converting frequency feature tensors to numpy array

    # Concatenate both feature sets along the second axis (column-wise)
    x = np.concatenate((sequence_features, frequency_features), axis=1)
    
    y = data['labels'].numpy()  # converting labels to numpy array
    return x, y

def classic_ml_training_loop(all_data_slices,
                            folds = 5,
                            use_aa_features = True,
                            metric_to_maximize = 'r2',
                            random_seed = 42, 
                            model_name = None,
                            model_to_use = "random"):

    all_results = {}

    for k,v in all_data_slices.items():
        
        sequences = v['full_dataset'][0]
        aa_features = v['full_dataset'][1]
        labels = v['full_dataset'][2]

        print(sequences.shape, ' --- sequences shape')
        print(aa_features.shape, ' --- aa_features shape')

        
        if not model_name:
            model_name = model_to_use + f'_{k}'
        else:
            model_name = model_name + f'_{k}'

        if use_aa_features:
            #flatten the dimensions
            sequences = sequences.reshape(sequences.shape[0], -1)  # Flatten each sequence to allow for aa_features 
            x_data = np.concatenate([sequences, aa_features], axis = 1)

        best_model, rmse, mae, r2 = train_with_cv_and_tuning(x = x_data, y = labels, random_seed = random_seed, model_to_use = model_to_use, n_splits = folds)

        all_results[k] = {'best_model':best_model, 'rmse':rmse, 'mae': mae, 'r2':r2}

    return all_results

    mae = np.mean(scores['test_mae']) 


def train_with_cv_and_tuning(x,
                            y,
                            random_seed = 42,
                            n_splits=5,
                            model_to_use = 'random_forest',
                            metric_to_maximize = 'r2', model_name = 'Generic_model'):
    # Initialize the Gradient Boosting Regressor model
    if model_to_use == 'random_forest':
        model = RandomForestRegressor(random_state=random_seed)
        # Parameters specific to Random Forest

        param_grid = {'max_depth':[15,20,25],
                        'n_estimators':[100,200,400],
                        'min_samples_split': [5, 10, 15]}
        
        

    elif model_to_use == 'gbm':
        model = GradientBoostingRegressor(random_state=random_seed)

        # Parameters specific to GBM
        # param_grid = {
        #     'n_estimators': [100, 200, 300, 400],
        #     'learning_rate': [0.005, 0.01, 0.05, 0.1],#lower learning rates typically work better with higher estimators
        #     'max_depth': [10,15,20,25], #increasing depth
        #     'min_samples_split': [20,30,50] #increase to combat overfitting
        # }
        param_grid = {'n_estimators': [200, 300, 400],
            'max_depth': [10,15,20], #increasing depth
        }

    # Define scorer to maximize R^2 score
    

    # Grid search with cross-validation, maximizing R^2
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=metric_to_maximize, cv=n_splits, verbose = 2)
    grid_search.fit(x, y)

    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_

    # Save the best model
    try:
        dump(best_model, f'{model_name}.joblib')
        print(f"Model saved as best_model.joblib.")
    except Exception as e:
        print(e, '---- issue in dummping model')

    # Compute scores using cross-validation on the best model
    comprehensive_scoring = {'mse': make_scorer(mean_squared_error, greater_is_better=False), 
                             'mae': make_scorer(mean_absolute_error, greater_is_better=False), 
                             'r2': make_scorer(r2_score)}
    scores = cross_validate(best_model, x, y, cv=n_splits, scoring=comprehensive_scoring, return_train_score=False)

    # Calculate mean of scores
    rmse = np.sqrt(-np.mean(scores['test_mse']))  
    mae = np.mean(scores['test_mae']) 
    r2 = np.mean(scores['test_r2'])

    print(f"Mean RMSE: {rmse}")
    print(f"Mean MAE: {mae}")
    print(f"Mean R^2: {r2}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    return best_model, rmse, mae, r2


class classic_ml_data_prep():
    def __init__(self):
        self.amino_acid_mapping = {
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

    def one_hot_encode(self, sequences):
    # One-hot encode a list of sequences
        max_length = max(len(seq) for seq in sequences)
        encoded_array = np.zeros((len(sequences), max_length, len(self.amino_acid_mapping)), dtype=np.float32)

        for idx, sequence in enumerate(sequences):
            for pos, aa in enumerate(sequence):
                if aa in self.amino_acid_mapping:
                    encoded_array[idx, pos, self.amino_acid_mapping[aa]] = 1.0
        return encoded_array

    def sequences_to_indices(self, sequences):
        # Convert a list of sequences to lists of indices
        indices_list = []
        for sequence in sequences:
            indices = [self.amino_acid_mapping[aa] for aa in sequence if aa in self.amino_acid_mapping]
            indices_list.append(indices)
        return indices_list
