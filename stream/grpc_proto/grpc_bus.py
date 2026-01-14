import argparse
import time
from concurrent import futures

import grpc

import stream_mic
from grpc_proto.audio_bus_pb2 import AudioFrame
from grpc_proto import audio_bus_pb2_grpc


class AudioBusService(audio_bus_pb2_grpc.AudioBusServicer):
    def __init__(self, cfg):
        self._cfg = cfg

    def Subscribe(self, request, context):
        sampling_rate = int(self._cfg.get("sampling_rate", self._cfg.get("sampling_rat")))
        channels = int(self._cfg.get("channels"))
        for payload in stream_mic.stream_audio(self._cfg):
            raw = payload["raw"]
            beam = payload["beam"]
            echo = payload["echo"]
            raw_channel_indices = payload.get("raw_channels", [])

            samples_per_channel = 0
            raw_channels = []
            raw_values = []
            if raw is not None:
                raw_channels = list(raw_channel_indices)
                samples_per_channel = raw.shape[1]
                raw_values = raw.astype("float32", copy=False).ravel().tolist()

            beam_values = []
            if beam is not None:
                samples_per_channel = samples_per_channel or beam.shape[0]
                beam_values = beam.astype("float32", copy=False).ravel().tolist()

            echo_values = []
            if echo is not None:
                samples_per_channel = samples_per_channel or echo.shape[0]
                echo_values = echo.astype("float32", copy=False).ravel().tolist()

            frame = AudioFrame(
                timestamp_unix_ms=int(time.time() * 1000),
                sampling_rate=sampling_rate,
                channels=channels,
                samples_per_channel=int(samples_per_channel),
                raw_channels=raw_channels,
                raw=raw_values,
                beam=beam_values,
                echo=echo_values,
            )
            yield frame


def serve(cfg_path, host, port):
    cfg = stream_mic._load_config(cfg_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    audio_bus_pb2_grpc.add_AudioBusServicer_to_server(AudioBusService(cfg), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"grpc bus listening on {host}:{port}")
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description="Publish raw/beam/echo audio over gRPC.")
    parser.add_argument("--config", default="configs/stream.yaml", help="Path to stream configuration")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind")
    args = parser.parse_args()

    serve(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
