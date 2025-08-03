import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """!
    @brief A double convolution block for the U-Net architecture.

    This class implements a block consisting of two convolutional layers, each followed by batch normalization,
    ReLU activation, and a dropout layer for regularization.

    @param in_channels (int): Number of input channels.
    @param out_channels (int): Number of output channels.
    @param dropout_rate (float, optional): Dropout probability for regularization (default: 0.1).
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Add dropout after the second ReLU
        )

    def forward(self, x):
        """!
        @brief Forward pass of the double convolution block.

        @param x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        @return torch.Tensor: Output tensor of shape [batch_size, out_channels, height, width].
        """
        return self.conv(x)

class UNET(nn.Module):
    """!
    @brief U-Net architecture for image segmentation.

    This class implements the U-Net model with downsampling (encoder), bottleneck, and upsampling (decoder) paths,
    including skip connections for improved feature retention.

    @param in_channels (int, optional): Number of input channels (default: 3).
    @param out_channels (int, optional): Number of output channels (default: 1).
    @param features (list, optional): List of feature sizes for each layer (default: [32, 64, 128, 256]).
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()  #!< List of upsampling layers.
        self.downs = nn.ModuleList()  #!< List of downsampling layers.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  #!< Max pooling layer for downsampling.
        
        # Downsampling part of U-Net
        for feature in features:
            dropout_rate = 0.1 if feature <= 128 else 0.2  # Higher rate for deeper layers
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature
            
        # Upsampling part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            dropout_rate = 0.1 if feature <= 128 else 0.2  # Higher rate for deeper layers
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate))
            
        # Bottleneck with higher dropout
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=0.3)  #!< Bottleneck layer.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  #!< Final convolution layer.
        
        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        """!
        @brief Forward pass of the U-Net model.

        Processes the input through the downsampling path, bottleneck, and upsampling path with skip connections.

        @param x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        @return torch.Tensor: Output tensor of shape [batch_size, out_channels, height, width].
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def initialize_weights(self):
        """!
        @brief Initializes the weights of convolutional and batch normalization layers.

        Uses Kaiming initialization for Conv2d and ConvTranspose2d layers, and constant initialization
        for BatchNorm2d layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:  # Initialize bias only if present
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def test():
    """!
    @brief Tests the U-Net model with a sample input.

    Creates a random input tensor and passes it through the U-Net model to verify output dimensions.
    Prints the shapes of the input and output tensors for debugging.
    """
    # x = torch.randn((1, 3, 256, 256))
    x = torch.randn((1, 3, 144, 256))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)  # Expected: [1, 1, 256, 256] ; [1, 1, 144, 256]
    print(x.shape)      # Expected: [1, 3, 256, 256] ; [1, 3, 144, 256]

if __name__ == "__main__":
    test()