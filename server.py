"""
GStreamer-based server for DepthSplat inference with dual camera streams.

This server:
- Accepts 2 simultaneous camera input streams via GStreamer
- Synchronizes frames from both cameras before processing
- Distributes workload across available GPUs
- Supports throttling and frame skipping
- Outputs PLY files in correct sequential order

GStreamer Documentation: https://gstreamer.freedesktop.org/documentation/?gi-language=c

Requirements:
    GStreamer and Python bindings must be installed:
    
    # Ubuntu/Debian:
    sudo apt-get install -y \\
        gstreamer1.0-tools \\
        gstreamer1.0-plugins-base \\
        gstreamer1.0-plugins-good \\
        gstreamer1.0-plugins-bad \\
        gstreamer1.0-plugins-ugly \\
        gstreamer1.0-libav \\
        python3-gst-1.0 \\
        gir1.2-gst-plugins-base-1.0

Usage:
    # Test mode (uses test patterns, no real cameras needed):
    python server.py --mode test
    
    # V4L2 cameras:
    python server.py --mode v4l2 --device0 /dev/video0 --device1 /dev/video2
    
    # RTSP streams:
    python server.py --mode rtsp --rtsp0 rtsp://... --rtsp1 rtsp://...
    
    # Custom (edit CONFIG in this file):
    python server.py --mode custom

Configuration:
    Modify the ServerConfig class or use command-line arguments.
    
Throttling/Skipping:
    --frame-skip N    Process every Nth frame (default: 1 = all frames)
    --max-fps F       Maximum processing rate in FPS (default: unlimited)
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

import threading
import queue
import time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import OrderedDict
import io

from inference import DepthSplatInference, InferenceConfig

# ============================================================================
# Configuration - Modify these settings as needed
# ============================================================================

@dataclass
class ServerConfig:
    """Server configuration settings."""
    
    # --- Camera Stream Settings ---
    # GStreamer pipeline sources for each camera
    # Examples:
    #   - Test pattern: "videotestsrc pattern=ball is-live=true ! videoconvert"
    #   - V4L2 camera: "v4l2src device=/dev/video0 ! videoconvert"
    #   - RTSP stream: "rtspsrc location=rtsp://... ! rtph264depay ! avdec_h264 ! videoconvert"
    #   - File: "filesrc location=video.mp4 ! decodebin ! videoconvert"
    camera_0_source: str = "videotestsrc pattern=ball is-live=true ! videoconvert"
    camera_1_source: str = "videotestsrc pattern=smpte is-live=true ! videoconvert"
    
    # Camera stream dimensions (input resolution)
    input_width: int = 1280
    input_height: int = 720
    input_framerate: int = 30
    
    # --- Processing Settings ---
    # Target resolution for inference (must match model expectations)
    target_width: int = 960
    target_height: int = 512
    
    # --- Throttling Settings ---
    # Frame skip: process every Nth frame (1 = process all, 2 = skip every other, etc.)
    frame_skip: int = 1
    
    # Maximum processing rate (frames per second, 0 = unlimited)
    max_fps: float = 0.0
    
    # Maximum queue size for pending frame pairs (older frames dropped when full)
    max_queue_size: int = 10
    
    # --- Duration/Limit Settings ---
    # Duration in seconds to run (0 = run indefinitely until Ctrl+C)
    duration_seconds: float = 0.0
    
    # Maximum number of frame pairs to process (0 = unlimited)
    max_frame_pairs: int = 0
    
    # --- GPU Settings ---
    # List of GPU device IDs to use (empty = auto-detect all available)
    gpu_ids: list = field(default_factory=list)
    
    # Number of worker threads per GPU
    workers_per_gpu: int = 1
    
    # --- Output Settings ---
    output_dir: str = "stream-output"
    
    # Callback for PLY output (if None, saves to files)
    # Signature: callback(sequence_id: int, ply_bytes: bytes)
    ply_callback: Optional[Callable[[int, bytes], None]] = None
    
    # --- Near/Far Plane Settings ---
    near_disparity: float = 1.0
    far_disparity: float = 0.1
    
    # --- Debug Settings ---
    verbose: bool = True
    save_input_frames: bool = False  # Save input frames for debugging
    
    # --- Sync Settings ---
    # Use frame count sync instead of timestamp sync (better for test sources)
    use_frame_count_sync: bool = False


# Default configuration instance
CONFIG = ServerConfig()


# ============================================================================
# Camera Metadata Configuration
# ============================================================================

# Default camera intrinsics and extrinsics for the two cameras
# These should be calibrated for your actual camera setup
# Format: intrinsics as 3x3 matrix, extrinsics as 4x4 camera-to-world matrix

DEFAULT_CAMERA_0_INTRINSICS = np.array([
    [500.0, 0.0, 640.0],
    [0.0, 500.0, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DEFAULT_CAMERA_0_EXTRINSICS = np.array([
    [1.0, 0.0, 0.0, -0.1],  # Slightly left of center
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

DEFAULT_CAMERA_1_INTRINSICS = np.array([
    [500.0, 0.0, 640.0],
    [0.0, 500.0, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DEFAULT_CAMERA_1_EXTRINSICS = np.array([
    [1.0, 0.0, 0.0, 0.1],  # Slightly right of center
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)


# ============================================================================
# Frame Synchronizer
# ============================================================================

class FrameSynchronizer:
    """
    Synchronizes frames from multiple camera streams.
    
    Pairs frames by timestamp proximity and ensures both cameras
    have contributed a frame before emitting a synchronized pair.
    
    For test sources, uses frame count matching instead of timestamps
    since videotestsrc timestamps may not align.
    """
    
    def __init__(self, num_cameras: int = 2, max_time_diff_ms: float = 100.0, use_frame_count_sync: bool = False):
        self.num_cameras = num_cameras
        self.max_time_diff_ms = max_time_diff_ms
        self.use_frame_count_sync = use_frame_count_sync
        self.buffers = {i: OrderedDict() for i in range(num_cameras)}
        self.frame_counts = {i: 0 for i in range(num_cameras)}
        self.lock = threading.Lock()
        self.sequence_counter = 0
        
    def add_frame(self, camera_id: int, timestamp_ns: int, frame: np.ndarray) -> Optional[tuple]:
        """
        Add a frame from a camera and attempt to find a synchronized pair.
        
        Args:
            camera_id: Camera index (0 or 1)
            timestamp_ns: Frame timestamp in nanoseconds
            frame: Frame data as numpy array [H, W, C]
            
        Returns:
            Tuple of (sequence_id, frames_dict) if a pair is found, None otherwise.
            frames_dict maps camera_id to (timestamp, frame) tuples.
        """
        with self.lock:
            # For frame count sync, use frame count as the key
            if self.use_frame_count_sync:
                key = self.frame_counts[camera_id]
                self.frame_counts[camera_id] += 1
            else:
                key = timestamp_ns
                
            self.buffers[camera_id][key] = (timestamp_ns, frame)
            
            # Try to find a matching pair
            result = self._try_match()
            
            # Clean up old frames (keep last 30 frames per camera)
            for cam_id in range(self.num_cameras):
                while len(self.buffers[cam_id]) > 30:
                    self.buffers[cam_id].popitem(last=False)
            
            return result
    
    def _try_match(self) -> Optional[tuple]:
        """Try to find a synchronized frame pair."""
        # Need frames from all cameras
        if any(len(self.buffers[i]) == 0 for i in range(self.num_cameras)):
            return None
        
        if self.use_frame_count_sync:
            # Match by frame count - find common frame counts
            keys_0 = set(self.buffers[0].keys())
            keys_1 = set(self.buffers[1].keys())
            common_keys = keys_0 & keys_1
            
            if not common_keys:
                return None
            
            # Use lowest common key
            match_key = min(common_keys)
            ts_0, frame_0 = self.buffers[0].pop(match_key)
            ts_1, frame_1 = self.buffers[1].pop(match_key)
            
            seq_id = self.sequence_counter
            self.sequence_counter += 1
            
            return (seq_id, {
                0: (ts_0, frame_0),
                1: (ts_1, frame_1)
            })
        else:
            # Match by timestamp proximity
            ref_key = next(iter(self.buffers[0]))
            ref_ts, ref_frame = self.buffers[0][ref_key]
            
            # Find closest frame from camera 1
            best_match_key = None
            best_diff = float('inf')
            
            for key, (ts, frame) in self.buffers[1].items():
                diff = abs(ts - ref_ts) / 1_000_000  # Convert to ms
                if diff < best_diff:
                    best_diff = diff
                    best_match_key = key
            
            # Check if match is within threshold
            if best_match_key is not None and best_diff <= self.max_time_diff_ms:
                # Remove matched frames from buffers
                _, frame_0 = self.buffers[0].pop(ref_key)
                ts_1, frame_1 = self.buffers[1].pop(best_match_key)
                
                seq_id = self.sequence_counter
                self.sequence_counter += 1
                
                return (seq_id, {
                    0: (ref_ts, frame_0),
                    1: (ts_1, frame_1)
                })
            
            return None


# ============================================================================
# GPU Worker Pool
# ============================================================================

class GPUWorker:
    """Worker that processes frame pairs on a specific GPU."""
    
    def __init__(self, gpu_id: int, worker_id: int, config: ServerConfig):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.config = config
        self.model: Optional[DepthSplatInference] = None
        self.task_queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()  # Signals when model is loaded
        
    def start(self):
        """Start the worker thread."""
        self.running = True
        self.ready_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"GPUWorker-{self.gpu_id}-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
    
    def wait_ready(self, timeout: float = None) -> bool:
        """Wait for the worker to be ready (model loaded)."""
        return self.ready_event.wait(timeout=timeout)
        
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.task_queue.put(None)  # Poison pill
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def submit(self, task: tuple):
        """Submit a task to this worker."""
        self.task_queue.put(task)
        
    def _run(self):
        """Worker thread main loop."""
        # Initialize model on this GPU
        torch.cuda.set_device(self.gpu_id)
        
        if self.config.verbose:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Initializing model on GPU {self.gpu_id}")
        
        self.model = DepthSplatInference()
        
        if self.config.verbose:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Ready")
        
        # Signal that we're ready
        self.ready_event.set()
        
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                    
                seq_id, frames_dict, result_callback = task
                self._process_task(seq_id, frames_dict, result_callback)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker {self.gpu_id}:{self.worker_id}] Error: {e}")
                
    def _process_task(self, seq_id: int, frames_dict: dict, result_callback: Callable):
        """Process a frame pair and invoke callback with result."""
        try:
            start_time = time.time()
            
            # Extract frames and convert to tensors
            frame_0 = frames_dict[0][1]  # (timestamp, frame) -> frame
            frame_1 = frames_dict[1][1]
            
            # Convert from HWC uint8 to CHW float [0, 1]
            frame_0_tensor = torch.from_numpy(frame_0).permute(2, 0, 1).float() / 255.0
            frame_1_tensor = torch.from_numpy(frame_1).permute(2, 0, 1).float() / 255.0
            
            # Stack frames
            images = torch.stack([frame_0_tensor, frame_1_tensor], dim=0)
            
            # Prepare camera data
            intrinsics_list = [
                DEFAULT_CAMERA_0_INTRINSICS,
                DEFAULT_CAMERA_1_INTRINSICS,
            ]
            extrinsics_list = [
                DEFAULT_CAMERA_0_EXTRINSICS,
                DEFAULT_CAMERA_1_EXTRINSICS,
            ]
            original_sizes = [
                (self.config.input_width, self.config.input_height),
                (self.config.input_width, self.config.input_height),
            ]
            
            # Run inference
            inference_config = InferenceConfig(
                target_height=self.config.target_height,
                target_width=self.config.target_width,
                near_disparity=self.config.near_disparity,
                far_disparity=self.config.far_disparity,
                verbose=False,  # Suppress verbose output in streaming mode
            )
            
            ply_bytes = self.model.run_from_data(
                images=images,
                intrinsics_list=intrinsics_list,
                extrinsics_list=extrinsics_list,
                original_sizes=original_sizes,
                image_filenames=[f"cam0_seq{seq_id}", f"cam1_seq{seq_id}"],
                config=inference_config,
            )
            
            elapsed = time.time() - start_time
            
            if self.config.verbose:
                print(f"[Worker {self.gpu_id}:{self.worker_id}] Processed seq {seq_id} in {elapsed:.3f}s")
            
            # Invoke callback with result
            result_callback(seq_id, ply_bytes)
            
        except Exception as e:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Failed to process seq {seq_id}: {e}")
            import traceback
            traceback.print_exc()


class GPUWorkerPool:
    """Pool of GPU workers for distributed inference."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.workers: list[GPUWorker] = []
        self.worker_index = 0
        self.lock = threading.Lock()
        
    def start(self):
        """Initialize and start all workers (non-blocking)."""
        # Determine available GPUs
        if self.config.gpu_ids:
            gpu_ids = self.config.gpu_ids
        elif torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            print("WARNING: No CUDA GPUs available, using CPU (will be slow)")
            gpu_ids = [0]  # Fake GPU ID for CPU fallback
        
        print(f"Initializing worker pool with GPUs: {gpu_ids}")
        
        for gpu_id in gpu_ids:
            for worker_idx in range(self.config.workers_per_gpu):
                worker = GPUWorker(gpu_id, worker_idx, self.config)
                self.workers.append(worker)
                worker.start()
        
        print(f"Worker pool started with {len(self.workers)} workers")
    
    def wait_all_ready(self, timeout: float = None) -> bool:
        """
        Wait for all workers to be ready (models loaded).
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if all workers are ready, False if timeout occurred
        """
        print(f"Waiting for {len(self.workers)} worker(s) to initialize...")
        start_time = time.time()
        
        for i, worker in enumerate(self.workers):
            remaining = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = max(0, timeout - elapsed)
                if remaining <= 0:
                    print(f"Timeout waiting for workers (only {i}/{len(self.workers)} ready)")
                    return False
            
            if not worker.wait_ready(timeout=remaining):
                print(f"Timeout waiting for worker {worker.gpu_id}:{worker.worker_id}")
                return False
        
        print(f"All {len(self.workers)} worker(s) ready!")
        return True
        
    def stop(self):
        """Stop all workers."""
        for worker in self.workers:
            worker.stop()
        print("Worker pool stopped")
        
    def submit(self, seq_id: int, frames_dict: dict, result_callback: Callable):
        """Submit a task to the next available worker (round-robin)."""
        with self.lock:
            worker = self.workers[self.worker_index]
            self.worker_index = (self.worker_index + 1) % len(self.workers)
        
        worker.submit((seq_id, frames_dict, result_callback))


# ============================================================================
# Output Sequencer
# ============================================================================

class OutputSequencer:
    """
    Ensures PLY outputs are emitted in correct sequential order.
    
    Buffers out-of-order results and releases them when all preceding
    sequences have been received.
    """
    
    def __init__(self, config: ServerConfig, on_complete: Optional[Callable[[int], None]] = None):
        self.config = config
        self.buffer: dict[int, bytes] = {}
        self.next_seq_to_emit = 0
        self.total_emitted = 0
        self.lock = threading.Lock()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_complete = on_complete  # Callback when a PLY is emitted
        
    def add_result(self, seq_id: int, ply_bytes: bytes):
        """
        Add a result and emit any ready sequences.
        
        Args:
            seq_id: Sequence ID of the result
            ply_bytes: PLY file contents
        """
        with self.lock:
            self.buffer[seq_id] = ply_bytes
            self._emit_ready()
            
    def _emit_ready(self):
        """Emit all ready sequences in order."""
        while self.next_seq_to_emit in self.buffer:
            seq_id = self.next_seq_to_emit
            ply_bytes = self.buffer.pop(seq_id)
            
            # Emit via callback or save to file
            if self.config.ply_callback:
                try:
                    self.config.ply_callback(seq_id, ply_bytes)
                except Exception as e:
                    print(f"PLY callback error for seq {seq_id}: {e}")
            else:
                # Save to file
                output_path = self.output_dir / f"output_{seq_id:06d}.ply"
                with open(output_path, 'wb') as f:
                    f.write(ply_bytes)
                
                if self.config.verbose:
                    size_mb = len(ply_bytes) / (1024 * 1024)
                    print(f"[Output] Saved seq {seq_id} to {output_path} ({size_mb:.2f} MB)")
            
            self.next_seq_to_emit += 1
            self.total_emitted += 1
            
            # Notify completion callback
            if self.on_complete:
                self.on_complete(self.total_emitted)


# ============================================================================
# GStreamer Camera Pipeline
# ============================================================================

class CameraPipeline:
    """GStreamer pipeline for a single camera stream."""
    
    def __init__(
        self,
        camera_id: int,
        source_pipeline: str,
        config: ServerConfig,
        frame_callback: Callable[[int, int, np.ndarray], None],
    ):
        """
        Initialize camera pipeline.
        
        Args:
            camera_id: Camera index (0 or 1)
            source_pipeline: GStreamer source pipeline string
            config: Server configuration
            frame_callback: Callback for received frames (camera_id, timestamp_ns, frame)
        """
        self.camera_id = camera_id
        self.source_pipeline = source_pipeline
        self.config = config
        self.frame_callback = frame_callback
        self.pipeline: Optional[Gst.Pipeline] = None
        self.frame_count = 0
        
    def build_pipeline(self) -> str:
        """Build the full GStreamer pipeline string."""
        # Source -> scale -> convert to RGB -> appsink
        width = self.config.input_width
        height = self.config.input_height
        framerate = self.config.input_framerate
        
        pipeline = (
            f"{self.source_pipeline} ! "
            f"videoscale ! "
            f"video/x-raw,width={width},height={height} ! "
            f"videorate ! "
            f"video/x-raw,framerate={framerate}/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=RGB ! "
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        return pipeline
        
    def start(self):
        """Start the camera pipeline."""
        pipeline_str = self.build_pipeline()
        
        if self.config.verbose:
            print(f"[Camera {self.camera_id}] Pipeline: {pipeline_str}")
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Connect appsink signal
        appsink = self.pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_new_sample)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start camera {self.camera_id} pipeline")
        
        if self.config.verbose:
            print(f"[Camera {self.camera_id}] Started")
            
    def stop(self):
        """Stop the camera pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            if self.config.verbose:
                print(f"[Camera {self.camera_id}] Stopped (processed {self.frame_count} frames)")
                
    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new frame from camera."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        # Apply frame skip
        self.frame_count += 1
        if self.config.frame_skip > 1:
            if (self.frame_count - 1) % self.config.frame_skip != 0:
                return Gst.FlowReturn.OK
        
        # Get buffer and timestamp
        buffer = sample.get_buffer()
        timestamp_ns = buffer.pts if buffer.pts != Gst.CLOCK_TIME_NONE else time.time_ns()
        
        # Extract frame data
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        try:
            # Convert to numpy array (RGB format)
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()  # Copy to own the data
            
            # Invoke callback
            self.frame_callback(self.camera_id, timestamp_ns, frame)
            
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK


# ============================================================================
# Main Server
# ============================================================================

class DepthSplatServer:
    """
    Main server coordinating camera streams, processing, and output.
    
    Usage:
        server = DepthSplatServer(config)
        server.start()
        # ... let it run ...
        server.stop()
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or CONFIG
        self.synchronizer = FrameSynchronizer(
            num_cameras=2,
            use_frame_count_sync=self.config.use_frame_count_sync
        )
        self.worker_pool: Optional[GPUWorkerPool] = None
        self.output_sequencer: Optional[OutputSequencer] = None
        self.cameras: list[CameraPipeline] = []
        self.main_loop: Optional[GLib.MainLoop] = None
        self.running = False
        
        # Rate limiting
        self.last_process_time = 0.0
        self.min_frame_interval = 1.0 / self.config.max_fps if self.config.max_fps > 0 else 0.0
        
        # Processing queue (with size limit)
        self.process_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.process_thread: Optional[threading.Thread] = None
        
        # Counters and timing
        self.start_time: float = 0.0
        self.frame_pairs_queued = 0
        self.frame_pairs_processed = 0
        self.processing_seq_counter = 0  # Sequential ID for frames that actually get processed
        self.seq_counter_lock = threading.Lock()
        
    def start(self):
        """Start the server."""
        # Initialize GStreamer
        Gst.init(None)
        
        print("="*70)
        print("Starting DepthSplat GStreamer Server")
        print("="*70)
        print(f"Input resolution: {self.config.input_width}x{self.config.input_height}")
        print(f"Target resolution: {self.config.target_width}x{self.config.target_height}")
        print(f"Frame skip: {self.config.frame_skip}")
        print(f"Max FPS: {self.config.max_fps if self.config.max_fps > 0 else 'unlimited'}")
        print(f"Max queue size: {self.config.max_queue_size}")
        print(f"Duration: {self.config.duration_seconds}s" if self.config.duration_seconds > 0 else "Duration: unlimited")
        print(f"Max frame pairs: {self.config.max_frame_pairs}" if self.config.max_frame_pairs > 0 else "Max frame pairs: unlimited")
        print(f"Sync mode: {'frame count' if self.config.use_frame_count_sync else 'timestamp'}")
        print("="*70)
        
        # Initialize components
        def on_output_complete(total_emitted: int):
            self.frame_pairs_processed = total_emitted
            # Check if we should stop after processing max_frame_pairs
            if self.config.max_frame_pairs > 0 and total_emitted >= self.config.max_frame_pairs:
                print(f"\n[Complete] Processed {total_emitted} frame pairs")
                self._request_stop()
        
        self.output_sequencer = OutputSequencer(self.config, on_complete=on_output_complete)
        self.worker_pool = GPUWorkerPool(self.config)
        self.worker_pool.start()
        
        # IMPORTANT: Wait for all workers to be ready before starting cameras
        # This ensures the encoder is fully loaded before we accept any frames
        print("\n" + "="*70)
        print("Waiting for encoder initialization...")
        print("="*70)
        if not self.worker_pool.wait_all_ready(timeout=300):  # 5 minute timeout
            print("ERROR: Workers failed to initialize within timeout")
            return
        print("="*70 + "\n")
        
        # Start processing thread
        self.running = True
        self.start_time = time.time()
        self.process_thread = threading.Thread(
            target=self._process_loop,
            name="ProcessLoop",
            daemon=True
        )
        self.process_thread.start()
        
        # Initialize camera pipelines - AFTER workers are ready
        self.cameras = [
            CameraPipeline(
                camera_id=0,
                source_pipeline=self.config.camera_0_source,
                config=self.config,
                frame_callback=self._on_frame,
            ),
            CameraPipeline(
                camera_id=1,
                source_pipeline=self.config.camera_1_source,
                config=self.config,
                frame_callback=self._on_frame,
            ),
        ]
        
        print("Starting camera pipelines...")
        for camera in self.cameras:
            camera.start()
        
        # Set up duration timeout if configured
        if self.config.duration_seconds > 0:
            def on_timeout():
                print(f"\n[Timer] Duration of {self.config.duration_seconds}s reached")
                self._request_stop()
                return False  # Don't repeat
            GLib.timeout_add(int(self.config.duration_seconds * 1000), on_timeout)
        
        # Set up status reporting
        def report_status():
            if not self.running:
                return False
            elapsed = time.time() - self.start_time
            print(f"[Status] Elapsed: {elapsed:.1f}s | Queued: {self.frame_pairs_queued} | Processed: {self.frame_pairs_processed}")
            return True  # Repeat
        GLib.timeout_add(5000, report_status)  # Report every 5 seconds
        
        # Run GLib main loop
        print("\nServer running. Press Ctrl+C to stop.\n")
        self.main_loop = GLib.MainLoop()
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def _request_stop(self):
        """Request the server to stop (can be called from any thread)."""
        if self.main_loop and self.main_loop.is_running():
            GLib.idle_add(self.main_loop.quit)
            
    def stop(self):
        """Stop the server."""
        print("\nStopping server...")
        
        self.running = False
        
        # Stop main loop
        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()
        
        # Stop cameras
        for camera in self.cameras:
            camera.stop()
        
        # Stop worker pool
        if self.worker_pool:
            self.worker_pool.stop()
        
        # Wait for process thread
        if self.process_thread:
            self.process_queue.put(None)  # Poison pill
            self.process_thread.join(timeout=5.0)
        
        print("Server stopped")
        
    def _on_frame(self, camera_id: int, timestamp_ns: int, frame: np.ndarray):
        """Handle incoming frame from a camera."""
        # Check if we should stop accepting new frames
        if not self.running:
            return
            
        # Check max_frame_pairs limit
        if self.config.max_frame_pairs > 0 and self.frame_pairs_queued >= self.config.max_frame_pairs:
            return
        
        # Try to synchronize with other camera
        result = self.synchronizer.add_frame(camera_id, timestamp_ns, frame)
        
        if result is not None:
            sync_seq_id, frames_dict = result
            
            # Apply rate limiting
            if self.min_frame_interval > 0:
                current_time = time.time()
                elapsed = current_time - self.last_process_time
                if elapsed < self.min_frame_interval:
                    # Silently skip - don't spam logs with throttle messages
                    return
                self.last_process_time = current_time
            
            # Assign a sequential processing ID (independent of sync seq_id)
            with self.seq_counter_lock:
                proc_seq_id = self.processing_seq_counter
                self.processing_seq_counter += 1
            
            # Queue for processing with the sequential processing ID
            try:
                self.process_queue.put_nowait((proc_seq_id, frames_dict))
                self.frame_pairs_queued += 1
                
                if self.config.verbose:
                    print(f"[Queue] Frame pair {proc_seq_id} queued (sync_seq={sync_seq_id})")
                
                # Check if we've reached the max and should stop accepting new frames
                if self.config.max_frame_pairs > 0 and self.frame_pairs_queued >= self.config.max_frame_pairs:
                    print(f"[Limit] Queued {self.config.max_frame_pairs} frame pairs, waiting for processing...")
                    
            except queue.Full:
                if self.config.verbose:
                    print(f"[Queue] Dropping frame pair (queue full)")
                    
    def _process_loop(self):
        """Process queued frame pairs."""
        while self.running:
            try:
                item = self.process_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    break
                    
                seq_id, frames_dict = item
                
                # Save input frames if debugging
                if self.config.save_input_frames:
                    self._save_debug_frames(seq_id, frames_dict)
                
                # Submit to worker pool
                self.worker_pool.submit(
                    seq_id,
                    frames_dict,
                    self.output_sequencer.add_result
                )
                
            except queue.Empty:
                continue
                
    def _save_debug_frames(self, seq_id: int, frames_dict: dict):
        """Save input frames for debugging."""
        debug_dir = Path(self.config.output_dir) / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        for cam_id, (timestamp, frame) in frames_dict.items():
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(debug_dir / f"seq{seq_id:06d}_cam{cam_id}.jpg")


# ============================================================================
# Utility Functions
# ============================================================================

def set_camera_calibration(
    camera_id: int,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
):
    """
    Set camera calibration for a specific camera.
    
    Args:
        camera_id: Camera index (0 or 1)
        intrinsics: 3x3 intrinsic matrix
        extrinsics: 4x4 camera-to-world matrix
    """
    global DEFAULT_CAMERA_0_INTRINSICS, DEFAULT_CAMERA_0_EXTRINSICS
    global DEFAULT_CAMERA_1_INTRINSICS, DEFAULT_CAMERA_1_EXTRINSICS
    
    if camera_id == 0:
        DEFAULT_CAMERA_0_INTRINSICS = intrinsics.astype(np.float32)
        DEFAULT_CAMERA_0_EXTRINSICS = extrinsics.astype(np.float32)
    elif camera_id == 1:
        DEFAULT_CAMERA_1_INTRINSICS = intrinsics.astype(np.float32)
        DEFAULT_CAMERA_1_EXTRINSICS = extrinsics.astype(np.float32)
    else:
        raise ValueError(f"Invalid camera_id: {camera_id}. Must be 0 or 1.")


def create_test_config(num_frames: int = 5, duration: float = 0.0) -> ServerConfig:
    """
    Create a test configuration using videotestsrc.
    
    Args:
        num_frames: Maximum number of frame pairs to process (0 = unlimited)
        duration: Duration in seconds (0 = use num_frames or unlimited)
    """
    return ServerConfig(
        # Use is-live=true for realistic timing behavior
        camera_0_source="videotestsrc pattern=ball is-live=true ! videoconvert",
        camera_1_source="videotestsrc pattern=smpte is-live=true ! videoconvert",
        input_width=640,
        input_height=480,
        input_framerate=10,
        frame_skip=1,      # Process all frames (use max_fps for rate control)
        max_fps=1.0,       # 1 frame pair per second
        max_frame_pairs=num_frames,
        duration_seconds=duration,
        use_frame_count_sync=True,  # Use frame count sync for test sources
        verbose=True,
    )


def create_v4l2_config(device_0: str = "/dev/video0", device_1: str = "/dev/video2") -> ServerConfig:
    """Create a configuration for V4L2 cameras."""
    return ServerConfig(
        camera_0_source=f"v4l2src device={device_0} ! videoconvert",
        camera_1_source=f"v4l2src device={device_1} ! videoconvert",
        input_width=1280,
        input_height=720,
        input_framerate=30,
        frame_skip=1,
        max_fps=10.0,
        verbose=True,
    )


def create_rtsp_config(url_0: str, url_1: str) -> ServerConfig:
    """Create a configuration for RTSP streams."""
    return ServerConfig(
        camera_0_source=f"rtspsrc location={url_0} latency=100 ! rtph264depay ! avdec_h264 ! videoconvert",
        camera_1_source=f"rtspsrc location={url_1} latency=100 ! rtph264depay ! avdec_h264 ! videoconvert",
        input_width=1920,
        input_height=1080,
        input_framerate=30,
        frame_skip=2,
        max_fps=15.0,
        verbose=True,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DepthSplat GStreamer Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode - process 5 frame pairs then exit:
  python server.py --mode test --num-frames 5
  
  # Test mode - run for 30 seconds:
  python server.py --mode test --duration 30
  
  # V4L2 cameras with throttling:
  python server.py --mode v4l2 --frame-skip 2 --max-fps 5
  
  # RTSP streams:
  python server.py --mode rtsp --rtsp0 rtsp://cam1/stream --rtsp1 rtsp://cam2/stream
"""
    )
    parser.add_argument(
        "--mode",
        choices=["test", "v4l2", "rtsp", "custom"],
        default="test",
        help="Camera mode (default: test)"
    )
    parser.add_argument(
        "--device0",
        default="/dev/video0",
        help="V4L2 device for camera 0"
    )
    parser.add_argument(
        "--device1",
        default="/dev/video2",
        help="V4L2 device for camera 1"
    )
    parser.add_argument(
        "--rtsp0",
        help="RTSP URL for camera 0"
    )
    parser.add_argument(
        "--rtsp1",
        help="RTSP URL for camera 1"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Process every Nth frame (default: mode-specific)"
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Maximum processing FPS (default: mode-specific)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frame pairs to process (0=unlimited, default: 5)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds (0=use num-frames, default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        default="stream-output",
        help="Output directory for PLY files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    # Create configuration based on mode
    if args.mode == "test":
        config = create_test_config(num_frames=args.num_frames, duration=args.duration)
    elif args.mode == "v4l2":
        config = create_v4l2_config(args.device0, args.device1)
    elif args.mode == "rtsp":
        if not args.rtsp0 or not args.rtsp1:
            parser.error("RTSP mode requires --rtsp0 and --rtsp1 URLs")
        config = create_rtsp_config(args.rtsp0, args.rtsp1)
    else:
        config = ServerConfig()
    
    # Apply command-line overrides (only if explicitly provided)
    if args.frame_skip is not None:
        config.frame_skip = args.frame_skip
    if args.max_fps is not None:
        config.max_fps = args.max_fps
    if args.duration > 0:
        config.duration_seconds = args.duration
    if args.num_frames > 0 and args.mode != "test":  # test mode already handles this
        config.max_frame_pairs = args.num_frames
        
    config.output_dir = args.output_dir
    config.verbose = not args.quiet
    
    # Start server
    server = DepthSplatServer(config)
    server.start()


if __name__ == "__main__":
    main()

