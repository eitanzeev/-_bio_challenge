import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn import preprocessing

class ProteinKd_CNN_Prediction(nn.Module):
    def __init__(self, **kwargs):
        super(ProteinKd_CNN_Prediction, self).__init__()
        self.config(kwargs)
        if self.use_cnn == True:
            self.build_model_CNN(kwargs)
        else:
            raise ValueError("Model type not supported")


    def build_model_CNN(self, kwargs):
        # Define layers
        if self.use_embedding:
            self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=self.embed_dim)
            input_channels = self.embed_dim
        else:
            input_channels = self.num_channels

        #NOTE include parameter to dynamically control channels outputs for conv1 layers
        if self.use_batch_norm:
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                nn.Flatten()
            )
        else:
            
            self.layers = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.conv_kernel_size, stride=self.conv_stride, padding=self.conv_padding),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride),
                nn.Flatten()
            )

        # Output size calculation
        output_size = self.calculate_output_size(self.sequence_length, self.layers)
        print(output_size, '---- output size prior to final calculation')
        final_input_size = (output_size) * 256 + (self.num_additional_features if self.use_aa_features else 0)

        print(final_input_size, '---- this is the final input size going into the linear layer')

        #We'll include a dropout layer that 
        if self.use_dropout:
            dropout = nn.Dropout(self.dropout_rate)
        else:
            dropout = nn.Identity()
        
        self.fullyconnected = nn.Sequential(
            nn.Linear(final_input_size, 128),
            nn.ReLU(),
            dropout,
            nn.Linear(128, 1)
        )

        print('---- {model_name} Model built ----'.format(model_name =self.model_name))
    
    def config(self, kwargs):

        self.use_cnn = kwargs.get('use_cnn', True)

        # Configurable features
        self.sequence_length = kwargs.get('sequence_length', 246)
        self.use_aa_features = kwargs.get('use_aa_features', False)
        self.num_additional_features = kwargs.get('num_additional_features', 0)

        self.embed_dim = kwargs.get('embed_dim', 64)
        self.use_embedding = kwargs.get('use_embedding', False)
        self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
        self.conv_stride = kwargs.get('conv_stride', 1)
        self.conv_padding = kwargs.get('conv_padding', 1) 
        self.pool_kernel_size = kwargs.get('pool_kernel_size', 2)
        self.pool_stride = kwargs.get('pool_stride', 2)
        self.num_channels = kwargs.get('num_channels', 20)
        self.conv_dilation = kwargs.get('conv_dilation', 1)
        
        self.use_dropout = kwargs.get('use_dropout', False)
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.use_batch_norm = kwargs.get('use_batch_norm', False)
        self.model_name = kwargs.get('model_name', 'Generic_ProteinKd')

    

    def forward(self, x, additional_features=None):
        
        if self.use_embedding:
            x = self.embedding(x)
            x = x.transpose(1,2)
        else:
            #print(x, ' ---- one hot encoding')
            #print(x.shape, ' ---- one hot encoding shape')
            x = x.float()
            x = x.transpose(1,2) # Change shape to (batch, channels, sequence_length)
        
        x = self.layers(x)

        
        #print(f"Shape after conv/pool layers: {x.shape}")  # Debugging line

        if self.use_aa_features and additional_features is not None:
            x = torch.cat((x, additional_features), dim=1)

            #print(f"Shape after concatenating additional features: {x.shape}")  # Debugging line

        x = self.fullyconnected(x)

        x = x.squeeze()
        return x

    def calculate_output_size(self, input_size, layers):
        output_size = input_size
        for pos, module in enumerate(layers):
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.MaxPool1d): #Do not apply to the relu or flatten layers
                    
                # Check if kernel_size, stride, and padding are tuples, take first element if so
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation



                if isinstance(module, nn.Conv1d):

                    effective_kernel_size = dilation * (kernel_size - 1) + 1
                    # print(effective_kernel_size, '--- this is the effective kernel size')

                    output_size = (((output_size + 2 * padding) - effective_kernel_size) // stride) + 1

                    # print(output_size, '---- output_size  conv1d layer {}'.format(pos))

                    # print("___________________________")

                if isinstance(module, nn.MaxPool1d):

                    output_size = ((output_size + 2 * padding - kernel_size) // stride) + 1

                    # print(output_size, '---- output_size maxpool layer {}'.format(pos))

                    #print("___________________________")


        return output_size