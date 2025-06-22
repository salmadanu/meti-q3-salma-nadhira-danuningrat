import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import Decoder, get_latent_dim

# Load decoder model
latent_dim = get_latent_dim("models/config.pth")
decoder = Decoder(latent_dim=latent_dim)
decoder.load_state_dict(torch.load("models/decoder.pth", map_location='cpu'))
decoder.eval()

# Helper function to one-hot encode label
def one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(torch.tensor(labels), num_classes=num_classes).float()

# Title and dropdown
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🖋️ Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate:", list(range(10)))

# Generate and show 5 images
if digit is not None:
    with torch.no_grad():
        z = torch.randn(5, latent_dim)  # 5 random latent vectors
        y = one_hot([digit] * 5)        # one-hot label for the digit
        generated = decoder(z, y).reshape(-1, 28, 28)  # shape: (5, 28, 28)

        # Plot the 5 images
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axs[i].imshow(generated[i].numpy(), cmap='gray')
            axs[i].axis('off')
        st.pyplot(fig)