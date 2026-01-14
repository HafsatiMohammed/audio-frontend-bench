import argparse

import grpc
import numpy as np

from grpc_proto.audio_bus_pb2 import SubscribeRequest
from grpc_proto import audio_bus_pb2_grpc


def _values_to_float32(values, channels, samples):
    if not values:
        return None
    array = np.array(values, dtype=np.float32)
    if channels and samples:
        return array.reshape(channels, samples)
    return array


def subscribe_audio_frames(host="127.0.0.1", port=50051):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = audio_bus_pb2_grpc.AudioBusStub(channel)
    for frame in stub.Subscribe(SubscribeRequest()):
        yield {
            "raw": _values_to_float32(frame.raw, len(frame.raw_channels), frame.samples_per_channel),
            "raw_channels": list(frame.raw_channels),
            "beam": _values_to_float32(frame.beam, 1 if frame.beam else 0, frame.samples_per_channel),
            "echo": _values_to_float32(frame.echo, 1 if frame.echo else 0, frame.samples_per_channel),
            "sampling_rate": frame.sampling_rate,
            "channels": frame.channels,
            "samples_per_channel": frame.samples_per_channel,
        }


def main():
    parser = argparse.ArgumentParser(description="Subscribe to the audio gRPC bus.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to read (0 = infinite)")
    args = parser.parse_args()

    frame_limit = args.frames
    for idx, payload in enumerate(subscribe_audio_frames(args.host, args.port), start=1):
        print(
            f"frame {idx}: raw={'yes' if payload['raw'] is not None else 'no'} "
            f"beam={'yes' if payload['beam'] is not None else 'no'} "
            f"echo={'yes' if payload['echo'] is not None else 'no'}"
        )
        if frame_limit and idx >= frame_limit:
            break


if __name__ == "__main__":
    main()
