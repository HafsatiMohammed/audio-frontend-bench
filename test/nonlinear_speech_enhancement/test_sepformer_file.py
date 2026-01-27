import torch
import numpy as np
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation

MODEL_ID = "speechbrain/sepformer-dns4-16k-enhancement"  # 16k enhancement model :contentReference[oaicite:1]{index=1}
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SepformerSeparation.from_hparams(source=MODEL_ID, savedir="pretrained_models/sepformer_dns4", run_opts={"device": device})

x, sr = sf.read("../../results_audio_files/linear_speech_enhancement/beam_sepformer-dns4-16k-enhancement.wav")
assert sr == 16000
x = x.astype(np.float32)
if x.ndim > 1:
    x = x[:, 0]  # force mono

xt = torch.from_numpy(x[None, :]).to(device)

with torch.inference_mode():
    est = model.separate_batch(xt)

print("est shape:", tuple(est.shape))

# Handle both common layouts:
# (B, T, S) or (B, S, T)
print(est.shape)
if est.ndim == 3 and est.shape[1] == xt.shape[1]:
    # (B, T, S)
    S = est.shape[2]
    y0 = est[0, :, 0].detach().cpu().numpy()
    sf.write("sepformer_out0.wav", y0, 16000)
    if S > 1:       
        y1 = est[0, :, 1].detach().cpu().numpy()
        sf.write("sepformer_out1.wav", y1, 16000)
else:
    # (B, S, T)
    S = est.shape[1]
    y0 = est[0, 0, :].detach().cpu().numpy()
    sf.write("sepformer_out0.wav", y0, 16000)
    if S > 1:
        y1 = est[0, 1, :].detach().cpu().numpy()
        sf.write("sepformer_out1.wav", y1, 16000)

print("Wrote sepformer_out0.wav" + (" and out1" if S > 1 else ""))



