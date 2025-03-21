from sklearn.base import is_classifier
import torch
import torch.nn as nn

from .utils import load_model

class MaxPoolBlock(nn.Module):
    def __init__(self, out_channels, kernel_size, padding, stride, ceil_mode):
        super().__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=ceil_mode, return_indices=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, max_indices = self.max_pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, max_indices

class MaxUnPoolBlock(nn.Module):
    def __init__(self, out_channels, kernel_size, padding, stride):
        super().__init__()

        self.max_unpool = nn.MaxUnpool2d(kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, max_indices, output_size):
        x = self.max_unpool(x, max_indices, output_size=output_size)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, encoded_dim, pooling_type, initial_out_channels=3, relu_slope=0, is_classifier=False, device='cpu'):
        super().__init__()
        
        self.is_classifier = is_classifier
        self.device = device
        
        # non parameterized functionality
        self.relu = nn.ReLU() # nn.LeakyReLU(relu_slope)
        self.flatten = nn.Flatten()
        
        out_channels = initial_out_channels
        
        # input.shape = in_channels x 176 x 144
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5),
                               stride=(2, 2), padding=(3, 3), bias=False)
        # height = (176 + 2*3 - 1*(5-1) - 1)/2 + 1 = 89
        # width = (144 + 2*3 - 1*(5-1) - 1)/2 + 1 = 73
        # output shape = (batch x out_channels x 89 x 73)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.max_indices = None
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), 
                                    padding=(1, 1), ceil_mode=False, return_indices=True)
        # height = ((89 + 2*1 - 1*(3-1) - 1)/2 + 1 = 45
        # width = ((73 + 2*1 - 1*(3-1) - 1)/2 + 1 = 37
        
        # first residual block
        self.block0 = self.res_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            pooling_type=pooling_type
        )
        # using this "block" to be able to return max_indices from the max pool layer
        self.max_pool_block0 = MaxPoolBlock(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            ceil_mode=True
        )
        # down sampling
        self.down_sample0 = self.down_sample_block(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        )

        out_channels *= 2
        
        self.block1 = self.res_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            pooling_type=pooling_type
        )
        self.max_pool_block1 = MaxPoolBlock(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            ceil_mode=True
        )
        self.down_sample1 = self.down_sample_block(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        )

        out_channels *= 2
        
        self.block2 = self.res_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            pooling_type=pooling_type
        )
        self.max_pool_block2 = MaxPoolBlock(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            ceil_mode=True
        )
        self.down_sample2 = self.down_sample_block(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        )
        
        out_channels *= 2
        
        # 240 * initial_out_channels
        self.encoded_mean = nn.Linear(240 * initial_out_channels, out_features=encoded_dim, bias=True)
        self.encoded_log_variance = nn.Linear(240 * initial_out_channels, out_features=encoded_dim, bias=True)
        
        self.final_out_channels = out_channels
        
    def res_block(self, out_channels, kernel_size, padding, stride, pooling_type):
        pooling_layer = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride) \
            if pooling_type == 'max' else nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
        
        return nn.Sequential(
            # bias is false as we have a bias in the BN layer
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu,
            pooling_layer,
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2)
        )
    
    def down_sample_block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
        )
      
    def forward(self, X):
        X0 = self.conv1(X)
        X0 = self.bn1(X0)
        X0 = self.relu(X0)
        X0, initial_max_indices = self.maxpool(X0)
        
        # residual block
        Y0_ = X0 + self.block0(X0)
        Y0 = self.down_sample0(Y0_)
        Y0 = self.relu(Y0)
        
        Y1_ = Y0 + self.block1(Y0)
        Y1 = self.down_sample1(Y1_)
        Y1 = self.relu(Y1)
        
        Y2_ = Y1 + self.block2(Y1)
        Y2 = self.down_sample2(Y2_)
        Y2 = self.relu(Y2)
        
        Y = self.flatten(Y2)

        if self.is_classifier:
            return Y
        
        encoded_mean = self.encoded_mean(Y)
        encoded_log_var = self.encoded_log_variance(Y)

        # using the reparameterization trick only when training the encoder
        epsilon = torch.randn_like(encoded_mean).to(dtype=torch.float32)
        encoded = encoded_mean + epsilon * torch.exp(encoded_log_var/2).to(dtype=torch.float32)
        
        return encoded, encoded_mean, encoded_log_var, initial_max_indices
    

class ConvDecoder(nn.Module):
    def __init__(self, encoded_dim, initial_out_channels, encoder_initial_out_channels, relu_slope=0):
        super().__init__()
        
        self.initial_out_channels = initial_out_channels
        out_channels = initial_out_channels
        
        self.relu = nn.ReLU() # nn.LeakyReLU(relu_slope)
        self.sigmoid = nn.Sigmoid()
        self.zero_padding = nn.ZeroPad2d(2)
        
        # 240 * encoder_initial_out_channels
        self.linear = nn.Linear(in_features=encoded_dim, out_features=240 * encoder_initial_out_channels, bias=True)
        
        out_channels = out_channels // 2

        # reshape to image shape before passing through this block
        self.up_sample0 = self.up_sample_block(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        )
        self.block0 = self.transpose_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )
        self.max_unpool_block0 = MaxUnPoolBlock(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )

        out_channels = out_channels // 2
        
        self.up_sample1 = self.up_sample_block(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        ) 
        self.block1 = self.transpose_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )
        self.max_unpool_block1 = MaxUnPoolBlock(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )
        
        out_channels = out_channels // 2
        
        self.up_sample2 = self.up_sample_block(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=2
        )
        self.block2 = self.transpose_block(
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1
        )
        self.max_unpool_block2 = MaxUnPoolBlock(
            out_channels=out_channels,
            kernel_size=(2, 2),
            padding=1,
            stride=1
        )
        
        self.max_indices = None
        self.max_unpool = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_transpose1 = nn.ConvTranspose2d(out_channels, 1, kernel_size=(5, 5),
                                                  stride=(2, 2), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        
    def transpose_block(self, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu,
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2)
        )
    
    def up_sample_block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
        )
    
    def set_max_indices(self, max_indices):
        self.max_indices = max_indices

    def forward(self, X):
        X0 = self.linear(X)
        # reshaping to (batch, channels, pixels) shape
        X0 = X0.reshape(-1, self.initial_out_channels, 6, 5)
        X0 = self.relu(X0)
        
        Y0_ = self.up_sample0(X0)
        Y0 = Y0_ + self.block0(Y0_)
        Y0 = self.relu(Y0)
        
        Y1_ = self.up_sample1(Y0)
        Y1 = Y1_ + self.block1(Y1_)
        Y1 = self.relu(Y1)
        
        Y2_ = self.up_sample2(Y1)
        Y2 = Y2_ + self.block2(Y2_)
        Y2 = self.relu(Y2)
        Y2 = self.zero_padding(Y2)
        
        Y = self.max_unpool(Y2, self.max_indices)
        num_pixels_to_cut = 1
        Y = self.conv_transpose1(Y)
        Y = self.bn1(Y)
        Y = self.relu(Y)
        # remove last row and column in the image
        Y = Y[..., :-num_pixels_to_cut, :-num_pixels_to_cut]

        # use sigmoid only if pixel values are in [0, 1] range
        # Y = self.sigmoid(Y)
        
        return Y
    
def kl_divergence_loss(mean, log_var):
    kl_sum = torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=1)
    return -0.5 * kl_sum.mean(dim=0)

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels, encoded_dim, initial_out_channels, relu_slope, device, pooling_type='max', is_classifier=False):
        super().__init__()

        self.device = device
        # latent space is encoded_dim dimensions
        self.encoder = ConvEncoder(in_channels=in_channels, encoded_dim=encoded_dim,
                                   initial_out_channels=initial_out_channels,
                                   pooling_type=pooling_type, relu_slope=relu_slope, is_classifier=is_classifier, device=device)
        self.decoder = ConvDecoder(encoded_dim=encoded_dim,
                                   initial_out_channels=self.encoder.final_out_channels,
                                   encoder_initial_out_channels=initial_out_channels, relu_slope=relu_slope)

    def forward(self, X):
        encoded, encoded_mean, encoded_log_var, max_indices = self.encoder(X)
        # when using only the decoder the last call of this line will be applied for every forward call in the decoder
        # limits usage in batches with the last batch size that was used in training
        self.decoder.set_max_indices(max_indices)
        decoded = self.decoder(encoded)
        
        return encoded, encoded_mean, encoded_log_var, decoded

    # def laplace_filtered(self, X):
    #     pass

class ConvClassifier(nn.Module):
    def __init__(self, train_params, num_classes, pooling_type='max', relu_slope=0, fine_tune=True, linear_blocks=True, device='cpu'):
        super().__init__()

        autoencoder = ConvAutoEncoder(in_channels=1, encoded_dim=train_params['encoded_dim'],
                                    initial_out_channels=train_params['initial_out_channels'],
                                    pooling_type=pooling_type, relu_slope=relu_slope, is_classifier=True, device=device)
        if fine_tune:
            autoencoder.load_state_dict(load_model(train_params))
        autoencoder.train()

        self.device = device
        
        self.encoder = autoencoder.encoder
        if fine_tune:
            # disabling autograd system to prevent retraining
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
        
        # relu_slope = 0 is the same as regular relu
        self.relu = nn.LeakyReLU(relu_slope)

        self.blocks = [
            self.linear_block(240 * train_params['initial_out_channels'], 256),
            # self.linear_block(512, 256),
            self.linear_block(256, 256),
            self.linear_block(256, 240 * train_params['initial_out_channels'])
        ] if linear_blocks else []
        self.seq_linear_blocks = nn.ModuleList(self.blocks)
        
        self.seq_classification_head = nn.Linear(240 * train_params['initial_out_channels'], num_classes)
        # self.seq_classification_head = nn.Linear(train_params['encoded_dim'], num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def linear_block(self, input_dim, output_dim, dropout_rate=0.25):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            self.relu,
            nn.Dropout(dropout_rate)
        )

    def forward(self, X):
        Y = self.encoder(X)

        for block in self.seq_linear_blocks:
            Y = block(Y)

        Y = self.seq_classification_head(Y)
        return self.log_softmax(Y)        
