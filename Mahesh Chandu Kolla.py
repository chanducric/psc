import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import os

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, image_size),
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability [0, 1] for real/fake
        )

    def forward(self, img):
        return self.model(img)

# Hyperparameters
latent_dim = 100  # Latent dimension for random noise
image_size = 784  # 28x28 images flattened into a 784-length vector
batch_size = 64
epochs = 100

# Create a folder to save generated images
os.makedirs("generated_images", exist_ok=True)

# Initialize the Generator and Discriminator
generator = Generator(latent_dim, image_size)
discriminator = Discriminator(image_size)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()  # Binary Cross Entropy loss
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Adam optimizer for generator
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Adam optimizer for discriminator

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.view(imgs.size(0), -1)  # Flatten the images for input to the discriminator

        # Labels for real and fake images
        valid = torch.ones((imgs.size(0), 1))  # Real label = 1
        fake = torch.zeros((imgs.size(0), 1))  # Fake label = 0

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn((imgs.size(0), latent_dim))  # Random noise input for generator
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)  # Generator's loss
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)  # Real image loss
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)  # Fake image loss (detach to avoid training generator)
        d_loss = (real_loss + fake_loss) / 2  # Total discriminator loss
        d_loss.backward()
        optimizer_D.step()

        # Print progress
        print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Save the generated images after each epoch
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim)
            gen_imgs = generator(z)
            gen_imgs = gen_imgs.view(16, 1, 28, 28)

        # Save generated images to disk
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        save_path = f"generated_images/epoch_{epoch}.png"
        torchvision.utils.save_image(grid, save_path)

        # Optionally, show the generated images
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Epoch {epoch}")
        plt.show()

# After training, generate and save new images
with torch.no_grad():
    z = torch.randn(16, latent_dim)  # Random noise
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.view(16, 1, 28, 28)

# Save final generated images
grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
torchvision.utils.save_image(grid, "generated_images/final_generated_images.png")

# Display the generated images
plt.imshow(grid.permute(1, 2, 0))
plt.title("Generated Images after Training")
plt.show()
