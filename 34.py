#NOTE: on this program, this was generated entirely by claude. Mostly to demonstrate git!



import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class OASISDataset(Dataset):
    """Dataset class for OASIS brain MRI data"""
    
    def __init__(self, data_dir, transform=None, slice_range=(50, 150)):
        """
        Args:
            data_dir (str): Directory containing OASIS preprocessed data
            transform (callable, optional): Optional transform to be applied on a sample
            slice_range (tuple): Range of slices to extract from 3D volumes (axial slices)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.slice_range = slice_range
        
        # Find all NIfTI files
        self.file_paths = list(self.data_dir.rglob("*.nii")) + list(self.data_dir.rglob("*.nii.gz"))
        
        if len(self.file_paths) == 0:
            print(f"No NIfTI files found in {data_dir}")
        else:
            print(f"Found {len(self.file_paths)} NIfTI files")
        
        # Pre-extract 2D slices from 3D volumes
        self.slices = []
        self._extract_slices()
    
    def _extract_slices(self):
        """Extract 2D slices from 3D volumes"""
        print("Extracting 2D slices from 3D volumes...")
        
        for file_path in tqdm(self.file_paths, desc="Processing volumes"):
            try:
                # Load NIfTI file
                nii_img = nib.load(str(file_path))
                volume = nii_img.get_fdata()
                
                # Extract axial slices within specified range
                start_slice = max(0, self.slice_range[0])
                end_slice = min(volume.shape[2], self.slice_range[1])
                
                for slice_idx in range(start_slice, end_slice):
                    slice_2d = volume[:, :, slice_idx]
                    
                    # Skip empty or mostly empty slices
                    if np.sum(slice_2d > 0) > 1000:  # Threshold for meaningful brain tissue
                        self.slices.append(slice_2d)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Extracted {len(self.slices)} 2D slices")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        slice_2d = self.slices[idx].copy()
        
        # Normalize to [0, 1]
        if slice_2d.max() > 0:
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        
        # Resize to standard size (adjust as needed)
        slice_2d = self._resize_slice(slice_2d, target_size=(128, 128))
        
        # Convert to tensor and add channel dimension
        slice_tensor = torch.FloatTensor(slice_2d).unsqueeze(0)
        
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        
        return slice_tensor
    
    def _resize_slice(self, slice_2d, target_size=(128, 128)):
        """Resize 2D slice to target size"""
        from scipy import ndimage
        zoom_factors = (target_size[0] / slice_2d.shape[0], 
                       target_size[1] / slice_2d.shape[1])
        return ndimage.zoom(slice_2d, zoom_factors, order=1)


class VAE(nn.Module):
    """Variational Autoencoder for brain MRI slices"""
    
    def __init__(self, input_channels=1, latent_dim=128, image_size=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128x128x1 -> 64x64x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 8x8x256 -> 4x4x512
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Calculate flattened dimension
        self.flattened_dim = 4 * 4 * 512  # 8192
        
        # Latent space
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_dim)
        
        self.decoder = nn.Sequential(
            # 4x4x512 -> 8x8x256
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32x64 -> 64x64x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64x64x32 -> 128x128x1
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent code to reconstruction"""
        z = self.fc_decode(z)
        z = z.view(z.size(0), 512, 4, 4)  # Reshape to feature maps
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence and reconstruction loss"""
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE, KLD


def train_vae(model, train_loader, val_loader, epochs=100, learning_rate=1e-3, 
              beta=1.0, device='cuda', save_path='brain_vae.pth'):
    """Train the VAE model"""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bce = 0
        train_kld = 0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bce += bce.item()
            train_kld += kld.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_bce = 0
        val_kld = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)
                
                val_loss += loss.item()
                val_bce += bce.item()
                val_kld += kld.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step()
        
        # Save model periodically
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return train_losses, val_losses


def visualize_results(model, test_loader, device='cuda', num_samples=8):
    """Visualize original and reconstructed images"""
    model.eval()
    
    with torch.no_grad():
        data = next(iter(test_loader))[:num_samples]
        data = data.to(device)
        
        recon_data, _, _ = model(data)
        
        # Move to CPU for visualization
        data = data.cpu()
        recon_data = recon_data.cpu()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(data[i, 0], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(recon_data[i, 0], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('vae_reconstruction_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def generate_samples(model, num_samples=8, device='cuda'):
    """Generate new samples from the latent space"""
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        
        samples = samples.cpu()
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(samples[i, 0], cmap='gray')
            axes[i].set_title(f'Generated Sample {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('vae_generated_samples.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main training function"""
    
    # Configuration
    data_dir = "/home/groups/comp3710/"  # Adjust this path as needed
    batch_size = 32
    latent_dim = 128
    epochs = 100
    learning_rate = 1e-3
    beta = 1.0  # KL divergence weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = OASISDataset(data_dir, slice_range=(50, 150))
    
    if len(dataset) == 0:
        print("No data found! Please check the data directory path.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = VAE(input_channels=1, latent_dim=latent_dim, image_size=128)
    
    # Train model
    train_losses, val_losses = train_vae(
        model, train_loader, val_loader, 
        epochs=epochs, learning_rate=learning_rate, 
        beta=beta, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Progress')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Visualize results
    visualize_results(model, test_loader, device)
    generate_samples(model, device=device)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()