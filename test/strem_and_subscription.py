import os
import sys
import threading
import time

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
STREAM_DIR = os.path.join(ROOT_DIR, "stream")
if STREAM_DIR not in sys.path:
    sys.path.insert(0, STREAM_DIR)

from grpc_proto import grpc_bus, grpc_subscribe


def launch_streaming():
    config_path = os.path.join(ROOT_DIR, "configs", "stream.yaml")
    def _start_bus():
        grpc_bus.serve(config_path, host="127.0.0.1", port=50051)

    server_thread = threading.Thread(target=_start_bus, daemon=True)
    server_thread.start()
    return server_thread


def test():
    launch_streaming()

    time.sleep(1.0)

    subscriber = grpc_subscribe.subscribe_audio_frames(host="127.0.0.1", port=50051)
    while True:
        payload = next(subscriber)
        raw = payload["raw"]
        beam = payload["beam"]
        echo = payload["echo"]
        print(raw.shape)
        print(beam.shape)
        print(echo.shape)

if __name__ == "__main__":
    test()
