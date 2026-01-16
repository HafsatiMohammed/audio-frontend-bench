import numpy as np
import onnxruntime


class DECStreamAEC:
    """
    Streaming wrapper around the ICASSP2022 DEC baseline ONNX model.

    Model settings match your offline code:
      - SR=16000
      - window_length=0.02 -> frame_size=320 (20ms)
      - hop_fraction=0.5 -> hop_size=160 (10ms)
      - dft_size=320 -> rfft bins = 161
      - hidden_size=322 -> h01/h02 shapes: (1,1,322)

    Streaming details:
      - maintains mic/far history buffer (frame_size)
      - runs inference every hop (10ms)
      - overlap-add output with sqrt Hann window
      - keeps an output FIFO so process_chunk returns exactly N samples each call
      - matches offline "pad_left=hop_size then return x_back[pad_left:]" by dropping first hop once
    """

    def __init__(
        self,
        model_path: str,
        sampling_rate: int = 16000,
        window_length: float = 0.02,
        hop_fraction: float = 0.5,
        dft_size: int = 320,
        hidden_size: int = 322,
        providers=None,
    ):
        self.sr = int(sampling_rate)
        self.window_length = float(window_length)
        self.hop_fraction = float(hop_fraction)
        self.dft_size = int(dft_size)
        self.hidden_size = int(hidden_size)

        self.frame_size = int(round(self.window_length * self.sr))          # 320
        self.hop_size = int(round(self.window_length * self.sr * self.hop_fraction))  # 160

        # sqrt Hann like your offline code
        w = np.hanning(self.frame_size + 1)[:-1]
        self.window = np.sqrt(w).astype(np.float32)

        # ONNX session
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = onnxruntime.InferenceSession(model_path, providers=providers)

        # recurrent states
        self.h01 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        self.h02 = np.zeros((1, 1, self.hidden_size), dtype=np.float32)

        # streaming buffers
        self.mic_hist = np.zeros(self.frame_size, dtype=np.float32)
        self.far_hist = np.zeros(self.frame_size, dtype=np.float32)

        self.ola = np.zeros(self.frame_size, dtype=np.float32)   # overlap-add buffer
        self.out_fifo = np.zeros((0,), dtype=np.float32)          # accumulated output samples

        # mimic offline pad_left=hop_size then drop left pad on return
        self.samples_to_drop = self.hop_size

    @staticmethod
    def _logpow(x: np.ndarray) -> np.ndarray:
        pspec = np.maximum(x * x, 1e-12)
        return np.log10(pspec)

    @staticmethod
    def _magphasor(cspec: np.ndarray):
        mspec = np.abs(cspec)
        pspec = np.empty_like(cspec)
        zero = mspec == 0.0
        pspec[zero] = 1.0
        pspec[~zero] = cspec[~zero] / mspec[~zero]
        return mspec, pspec

    def _calc_features(self, xmag_mic: np.ndarray, xmag_far: np.ndarray) -> np.ndarray:
        feat_mic = self._logpow(xmag_mic)
        feat_far = self._logpow(xmag_far)
        feat = np.concatenate([feat_mic, feat_far], axis=0)
        feat = (feat / 20.0).astype(np.float32, copy=False)
        # shape expected: (1,1,322)
        return feat[np.newaxis, np.newaxis, :]

    def reset(self):
        self.h01.fill(0.0)
        self.h02.fill(0.0)
        self.mic_hist.fill(0.0)
        self.far_hist.fill(0.0)
        self.ola.fill(0.0)
        self.out_fifo = np.zeros((0,), dtype=np.float32)
        self.samples_to_drop = self.hop_size

    def _push_hop(self, mic_hop: np.ndarray, far_hop: np.ndarray):
        """
        Process one hop (10ms) and append hop output to out_fifo.
        mic_hop/far_hop: (hop_size,) float32
        """
        # update history buffers
        hs = self.hop_size
        fs = self.frame_size

        self.mic_hist[:-hs] = self.mic_hist[hs:]
        self.mic_hist[-hs:] = mic_hop

        self.far_hist[:-hs] = self.far_hist[hs:]
        self.far_hist[-hs:] = far_hop

        # STFT-like step
        mic_win = self.mic_hist * self.window
        far_win = self.far_hist * self.window

        cspec_mic = np.fft.rfft(mic_win, n=self.dft_size)
        xmag_mic, xphs_mic = self._magphasor(cspec_mic)

        cspec_far = np.fft.rfft(far_win, n=self.dft_size)
        xmag_far = np.abs(cspec_far)

        feat = self._calc_features(xmag_mic, xmag_far)

        inputs = {"input": feat, "h01": self.h01, "h02": self.h02}
        mask, self.h01, self.h02 = self.sess.run(None, inputs)

        mask = mask[0, 0].astype(np.float32, copy=False)  # (161,)
        y_frame = np.fft.irfft(mask * xmag_mic * xphs_mic, n=self.dft_size).astype(np.float32, copy=False)
        y_frame *= self.window

        # overlap-add
        self.ola += y_frame[:fs]

        hop_out = self.ola[:hs].copy()

        # shift ola buffer
        self.ola[:-hs] = self.ola[hs:]
        self.ola[-hs:] = 0.0

        # drop initial pad_left (offline returns x_back[hop_size:])
        if self.samples_to_drop > 0:
            d = min(self.samples_to_drop, hop_out.shape[0])
            hop_out = hop_out[d:]
            self.samples_to_drop -= d

        if hop_out.size:
            self.out_fifo = np.concatenate([self.out_fifo, hop_out], axis=0)

    def process_chunk(self, mic: np.ndarray, far: np.ndarray) -> np.ndarray:
        """
        mic/far: (N,) or (1,N) float32, assumed 16kHz mono.
        returns: (N,) float32, same length as input (FIFO-padded if needed).
        """
        mic = np.asarray(mic, dtype=np.float32)
        far = np.asarray(far, dtype=np.float32)

        if mic.ndim == 2:
            mic = mic[0]
        if far.ndim == 2:
            far = far[0]

        N = int(min(mic.shape[0], far.shape[0]))
        mic = mic[:N]
        far = far[:N]

        # ensure we process in hop_size steps
        hs = self.hop_size
        pad = (-N) % hs
        if pad:
            mic = np.pad(mic, (0, pad))
            far = np.pad(far, (0, pad))

        # hop loop
        for i in range(0, mic.shape[0], hs):
            self._push_hop(mic[i:i+hs], far[i:i+hs])

        # return exactly N samples (use FIFO)
        if self.out_fifo.shape[0] >= N:
            out = self.out_fifo[:N].copy()
            self.out_fifo = self.out_fifo[N:]
        else:
            # not enough yet (only happens at very start); pad with zeros
            out = np.pad(self.out_fifo, (0, N - self.out_fifo.shape[0]))
            self.out_fifo = np.zeros((0,), dtype=np.float32)

        return out

    def flush(self, max_samples: int = 10_000) -> np.ndarray:
        """
        Drain remaining FIFO + OLA tail (best-effort).
        """
        # generate some tail by pushing zeros hops
        hs = self.hop_size
        z = np.zeros((hs,), dtype=np.float32)
        produced_before = self.out_fifo.shape[0]

        # push a few silent hops to flush overlap
        for _ in range(10):
            self._push_hop(z, z)

        out = self.out_fifo[:max_samples].copy()
        self.out_fifo = self.out_fifo[out.shape[0]:]
        return out
