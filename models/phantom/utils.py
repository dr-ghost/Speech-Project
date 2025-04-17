import os
import gdown
import torch

# from ._transformer import PhantomTransformer

def exists(obj):
    return obj is not None

def download_cpkt():
    """
    Download the wavlm_base_plus checkpoint from Google Drive.
    """
    file_ids = [
        "1OMdkp5Vv8A9WnHSTSoDwA8hxQWsEAu85",
    ]
    
    # Desired output file names
    file_names = ["phantom_wavlm_base_plus.pt"]
    
    # Google Drive direct download URL base format
    base_url = "https://drive.google.com/uc?export=download&id="
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    download_dir = os.path.join(current_dir, "checkpoints")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    url = base_url + file_ids[0]
    output_path = os.path.join(download_dir, file_names[0])
    
    print(f"Downloading {file_names[0]} from {url}...")
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded {file_names[0]} to {output_path}")
    
def ready_made_model():
    download_cpkt()
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(current_dir, "checkpoints")
        
        output_path = os.path.join(download_dir, "phantom_wavlm_base_plus.pt")
        
        ph = PhantomTransformer()
        ph.load_state_dict(torch.load(output_path))
    except:
        return None