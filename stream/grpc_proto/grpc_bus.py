import argparse
import sys
import time
import threading
import queue
from concurrent import futures

import grpc

import stream_mic
from grpc_proto.audio_bus_pb2 import AudioFrame
from grpc_proto import audio_bus_pb2_grpc


class SharedAudioStream:
    """Manages a single audio stream broadcast to all subscribers."""
    def __init__(self, cfg):
        self._cfg = cfg
        self._subscribers = []  # List of queues, one per subscriber
        self._error = None
        self._stream_thread = None
        self._lock = threading.Lock()
        self._active = False
        
    def start(self):
        """Start the audio stream in a background thread."""
        with self._lock:
            if self._active:
                return  # Already started
            self._active = True
            self._error = None
            
        def _stream_worker():
            try:
                for payload in stream_mic.stream_audio(self._cfg):
                    if not self._active:
                        break
                    # Broadcast to all subscribers - each has their own queue
                    with self._lock:
                        active_subscribers = [q for q in self._subscribers if q is not None]
                        if not active_subscribers:
                            continue  # No subscribers, skip this frame
                        for sub_queue in active_subscribers:
                            try:
                                # Use put_nowait with unbounded queue - should never fail
                                # But if subscriber is extremely slow, this will block briefly
                                sub_queue.put_nowait(("frame", payload))
                            except queue.Full:
                                # Should never happen with unbounded queue, but handle gracefully
                                print("Warning: Queue full for subscriber (should not happen with unbounded queue)", file=sys.stderr)
                                # Try to make room by removing oldest frame
                                try:
                                    sub_queue.get_nowait()
                                    sub_queue.put_nowait(("frame", payload))
                                except:
                                    pass
            except Exception as e:
                import sys
                print(f"Error in audio stream: {e}", file=sys.stderr)
                self._error = e
                # Broadcast error to all subscribers - each has their own queue
                with self._lock:
                    for sub_queue in self._subscribers:
                        if sub_queue is not None:
                            try:
                                sub_queue.put_nowait(("error", e))
                            except queue.Full:
                                # Should not happen, but try to make room
                                try:
                                    sub_queue.get_nowait()
                                    sub_queue.put_nowait(("error", e))
                                except:
                                    pass
            finally:
                # Signal all subscribers that stream is done - each has their own queue
                with self._lock:
                    for sub_queue in self._subscribers:
                        if sub_queue is not None:
                            try:
                                sub_queue.put_nowait(("done", None))
                            except queue.Full:
                                # Should not happen, but try to make room
                                try:
                                    sub_queue.get_nowait()
                                    sub_queue.put_nowait(("done", None))
                                except:
                                    pass
                    self._active = False
        
        self._stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        self._stream_thread.start()
    
    def subscribe(self):
        """Subscribe to the stream. Returns a queue that will receive frames."""
        # Use unbounded queue (maxsize=0) so frames are never dropped
        # Each subscriber gets their own independent queue
        sub_queue = queue.Queue(maxsize=0)  # Unbounded - will block if subscriber is very slow
        with self._lock:
            self._subscribers.append(sub_queue)
        return sub_queue
    
    def unsubscribe(self, sub_queue):
        """Unsubscribe from the stream."""
        with self._lock:
            if sub_queue in self._subscribers:
                self._subscribers.remove(sub_queue)
    
    def stop(self):
        """Stop the audio stream."""
        with self._lock:
            self._active = False
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)


class AudioBusService(audio_bus_pb2_grpc.AudioBusServicer):
    _shared_stream = None
    _stream_lock = threading.Lock()
    
    def __init__(self, cfg):
        self._cfg = cfg
        # Initialize shared stream if not already done
        with AudioBusService._stream_lock:
            if AudioBusService._shared_stream is None:
                AudioBusService._shared_stream = SharedAudioStream(cfg)
                AudioBusService._shared_stream.start()

    def Subscribe(self, request, context):
        """Subscribe to the shared audio stream."""
        sampling_rate = int(self._cfg.get("sampling_rate", self._cfg.get("sampling_rat")))
        channels = int(self._cfg.get("channels"))
        
        # Get a subscription queue
        sub_queue = AudioBusService._shared_stream.subscribe()
        frames_yielded = 0
        
        try:
            while context.is_active():
                try:
                    msg_type, payload = sub_queue.get(timeout=0.1)
                    if msg_type == "error":
                        # Stream error occurred
                        import sys
                        print(f"Subscribe: Broadcasting error to client: {payload}", file=sys.stderr)
                        raise payload
                    elif msg_type == "done":
                        # Stream ended
                        import sys
                        print(f"Subscribe: Stream done signal received, frames yielded: {frames_yielded}", file=sys.stderr)
                        break
                    elif msg_type == "frame":
                        # Process and yield the frame
                        try:
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
                            frames_yielded += 1
                        except Exception as e:
                            import sys
                            print(f"Subscribe: Error processing frame (frames_yielded: {frames_yielded}): {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            # Continue to next frame instead of breaking the connection
                            continue
                except queue.Empty:
                    # Timeout - check if context is still active and continue
                    if not context.is_active():
                        import sys
                        print(f"Subscribe: Context became inactive, frames yielded: {frames_yielded}", file=sys.stderr)
                        break
                    continue
        except Exception as e:
            # Log the error - this will be sent to the client
            import sys
            print(f"Error in Subscribe for client (frames yielded: {frames_yielded}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            # Unsubscribe when client disconnects
            import sys
            print(f"Subscribe: Client disconnected, total frames yielded: {frames_yielded}", file=sys.stderr)
            AudioBusService._shared_stream.unsubscribe(sub_queue)


def serve(cfg_path, host, port):
    cfg = stream_mic._load_config(cfg_path)
    # Configure server with longer timeouts to handle slow clients
    options = [
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300000),
        ('grpc.max_connection_idle_ms', 300000),  # 5 minutes
        ('grpc.max_connection_age_ms', 3600000),  # 1 hour
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
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
