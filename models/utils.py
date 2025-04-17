import torch
from .wavlm import wavlm_large, WavLM

from functools import partial
import warnings
warnings.filterwarnings("ignore")

SPEAKER_INFORMATION_LAYER = 6

def wavlm_embedding(wavlm: WavLM, wav: torch.Tensor) -> torch.Tensor:
    return wavlm.extract_features(wav, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0].squeeze_(0)

def wavlm_func_gen():
    wavlm_model = wavlm_large()
    wavlm_model.eval()
    for param in wavlm_model.parameters():
        param.requires_grad = False
    
    # return torch.compile(partial(wavlm_embedding, wavlm_model))

if __name__ == "__main__":
    wavlm_model = wavlm_large()
    wavlm_model.eval()
    for param in wavlm_model.parameters():
        param.requires_grad = False
    
    # Test the function
    x = torch.randn(1, 16000 * 5).to("cuda")  # Simulated audio input
    embedding = wavlm_embedding(wavlm_model, x)
    print(embedding.shape)  # Should print the shape of the embedding