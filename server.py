"""
GStreamer-based server for DepthSplat inference with dual camera streams.

This server:
- Accepts 2 simultaneous camera input streams via GStreamer
- Uses DepthAnything3 to estimate camera intrinsics and extrinsics (poses)
- Synchronizes frames from both cameras before processing
- Distributes workload across available GPUs
- Supports throttling and frame skipping
- Outputs PLY/SPZ files in correct sequential order
- Supports WebSocket streaming to remote clients (Mac, Vision Pro, web)

Pipeline:
    1. Camera frames captured via GStreamer (or loaded from files in test mode)
    2. DepthAnything3 estimates intrinsics and W2C extrinsics from frames
    3. W2C extrinsics are converted to C2W poses for DepthSplat
    4. Intrinsics are scaled from DepthAnything3's processed size to original input size
    5. DepthSplat encoder generates Gaussian splats using the original images + estimated cameras

Note on terminology:
    - DepthAnything3 outputs "extrinsics" in W2C (world-to-camera) format
    - DepthSplat's "extrinsics" parameter actually expects C2W (camera-to-world) poses
    - We convert W2C → C2W by taking the matrix inverse

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
    # File-based test mode (recommended for testing, no GStreamer needed):
    python server.py --mode file --image-dir /workspace/input_images
    
    # GStreamer test mode (uses test patterns, no real cameras needed):
    python server.py --mode test
    
    # V4L2 cameras:
    python server.py --mode v4l2 --device0 /dev/video0 --device1 /dev/video2
    
    # RTSP streams:
    python server.py --mode rtsp --rtsp0 rtsp://... --rtsp1 rtsp://...
    
    # RTMP streams:
    python server.py --mode rtmp --rtmp0 rtmp://server/live/stream1 --rtmp1 rtmp://server/live/stream2
    
    # SRT streams (listener mode - iOS devices send to server):
    python server.py --mode srt --srt-listen
    
    # TCP H.264 streams (listener mode - wait for raw H.264 senders):
    python server.py --mode tcp --tcp-listen
    
    # TCP H.264 streams (client mode - connect to senders):
    python server.py --mode tcp --tcp0 192.168.1.100:5000 --tcp1 192.168.1.100:5001
    
    # WebRTC mode (receive streams from browsers/mobile apps):
    python server.py --mode webrtc --webrtc-port 8080
    
    # Custom (edit CONFIG in this file):
    python server.py --mode custom

Configuration:
    Modify the ServerConfig class or use command-line arguments.
    
Throttling/Skipping:
    --frame-skip N    Process every Nth frame (default: 1 = all frames)
    --max-fps F       Maximum processing rate in FPS (default: unlimited)

WebRTC Input Mode:
    Receive camera streams from browsers or mobile apps via WebRTC:
    
    # Start server in WebRTC mode
    python server.py --mode webrtc --webrtc-port 8080
    
    The server provides:
    - HTTP server for signaling at http://server_ip:8080
    - WebSocket signaling endpoint at ws://server_ip:8080/ws
    - Optional STUN server configuration via --stun-server
    
    Clients should:
    1. Connect to the WebSocket signaling endpoint
    2. Send JSON: {"type": "register", "camera_id": 0}  (or 1 for second camera)
    3. Exchange SDP offers/answers and ICE candidates
    4. Stream video via WebRTC (H.264/VP8/VP9 supported)
    
    Message format (JSON):
    - Register: {"type": "register", "camera_id": 0}
    - SDP Offer: {"type": "offer", "sdp": "..."}
    - SDP Answer: {"type": "answer", "sdp": "..."} (from server)
    - ICE Candidate: {"type": "ice", "candidate": "...", "sdpMLineIndex": N}

WebSocket Output Streaming:
    Enable with --websocket flag to stream outputs to remote clients:
    
    # File mode with WebSocket streaming (stream + save files)
    python server.py --mode file --image-dir /path/to/images --websocket
    
    # File mode with WebSocket only (no file saving)
    python server.py --mode file --image-dir /path/to/images --websocket --ws-only
    
    # GStreamer mode with WebSocket streaming
    python server.py --mode test --websocket --ws-port 8765
    
    # GStreamer mode with WebSocket only
    python server.py --mode test --websocket --ws-only
    
    Clients connect to ws://server_ip:8765 and receive binary messages:
        [4 bytes: seq_id][4 bytes: format_type][4 bytes: length][N bytes: SPZ/PLY data]
    
    See websocket_streamer.py for full client implementation guide.
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import Gst, GstApp, GLib, GstWebRTC, GstSdp

import sys
import threading
import queue
import time
import asyncio
import json
import ssl
import subprocess
import os
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from collections import OrderedDict
import io
from PIL import Image

os.environ["HF_HOME"] = "/root/.cache"

# ============================================================================
# Import DepthSplat inference from ../depthsplat
# ============================================================================

# Add depthsplat directory to Python path
_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEPTHSPLAT_DIR = (_SCRIPT_DIR.parent / "depthsplat").resolve()

if not _DEPTHSPLAT_DIR.exists():
    raise ImportError(
        f"DepthSplat directory not found at: {_DEPTHSPLAT_DIR}\n"
        f"Expected directory structure:\n"
        f"  {_SCRIPT_DIR.parent}/\n"
        f"    ├── streamserver/  (this server)\n"
        f"    └── depthsplat/    (DepthSplat inference code)"
    )

sys.path.insert(0, str(_DEPTHSPLAT_DIR))

# Import the inference module and reconfigure paths to be absolute
import inference as _depthsplat_inference

# Make paths absolute relative to depthsplat directory
_depthsplat_inference.CHECKPOINT_PATH = str(_DEPTHSPLAT_DIR / _depthsplat_inference.CHECKPOINT_PATH)
_depthsplat_inference.CONFIG_ROOT = str(_DEPTHSPLAT_DIR / "config")

from inference import DepthSplatInference, InferenceConfig


# ============================================================================
# Import DepthAnything3 for camera intrinsics/extrinsics estimation
# ============================================================================

_DEPTH_ANYTHING_DIR = (_SCRIPT_DIR.parent / "Depth-Anything-3" / "src").resolve()

if not _DEPTH_ANYTHING_DIR.exists():
    raise ImportError(
        f"Depth-Anything-3 directory not found at: {_DEPTH_ANYTHING_DIR}\n"
        f"Expected directory structure:\n"
        f"  {_SCRIPT_DIR.parent}/\n"
        f"    ├── streamer/       (this server)\n"
        f"    ├── depthsplat/     (DepthSplat inference code)\n"
        f"    └── Depth-Anything-3/src/  (DepthAnything3 code)"
    )

sys.path.insert(0, str(_DEPTH_ANYTHING_DIR))

from depth_anything_3.api import DepthAnything3

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
    
    # Output format: "spz" (compressed, ~10x smaller) or "ply" (standard)
    output_format: str = "spz"
    
    # Callback for output (called in addition to file saving if save_files=True)
    # Signature: callback(sequence_id: int, output_bytes: bytes)
    output_callback: Optional[Callable[[int, bytes], None]] = None
    
    # Whether to save output files to disk (set False for streaming-only mode)
    save_files: bool = True
    
    # --- Near/Far Plane Settings ---
    near_disparity: float = 1.0
    far_disparity: float = 0.1
    
    # --- Debug Settings ---
    verbose: bool = True
    save_input_frames: bool = False  # Save input frames for debugging
    
    # Directory to save camera input frames before inference (None = don't save)
    # When set, frames are saved as camera_input_dir/seq_XXXXX_cam0.png and seq_XXXXX_cam1.png
    camera_input_dir: Optional[str] = None
    
    # --- Sync Settings ---
    # Use frame count sync instead of timestamp sync (better for test sources)
    use_frame_count_sync: bool = False
    
    # --- DepthAnything3 Settings ---
    # DepthAnything3 model name for camera estimation
    depth_anything_model: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    
    # Processing resolution for DepthAnything3 (504 is default)
    depth_anything_process_res: int = 504
    
    # --- WebRTC Settings ---
    # Port for WebRTC signaling server (HTTP + WebSocket)
    webrtc_port: int = 8080
    
    # Port for serving the WebRTC client HTML page
    web_client_port: int = 8888
    
    # STUN server for ICE (empty = use Google's public server)
    stun_server: str = "stun://stun.l.google.com:19302"
    
    # --- SSL/TLS Settings ---
    # Path to SSL certificate file (enables HTTPS/WSS when set)
    ssl_cert: Optional[str] = None
    
    # Path to SSL private key file
    ssl_key: Optional[str] = None
    
    # --- File-based Test Mode ---
    # When set, load images from this directory instead of cameras
    # Images should be named with _metadata.json sidecar files
    test_images_dir: Optional[str] = None


# Default configuration instance
CONFIG = ServerConfig()


# ============================================================================
# Camera Pose Utilities
# ============================================================================

def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """
    Convert world-to-camera (W2C) matrix to camera-to-world (C2W) matrix.
    
    DepthAnything3 outputs W2C (extrinsics), but DepthSplat expects C2W (poses).
    
    Args:
        w2c: World-to-camera matrix [N, 3, 4] or [N, 4, 4] or [3, 4] or [4, 4]
        
    Returns:
        Camera-to-world matrix [N, 4, 4] or [4, 4]
    """
    # Ensure 4x4 shape
    if w2c.ndim == 2:
        if w2c.shape == (3, 4):
            w2c_4x4 = np.eye(4, dtype=w2c.dtype)
            w2c_4x4[:3, :4] = w2c
            w2c = w2c_4x4
        c2w = np.linalg.inv(w2c).astype(np.float32)
    elif w2c.ndim == 3:
        n = w2c.shape[0]
        if w2c.shape[1:] == (3, 4):
            w2c_4x4 = np.zeros((n, 4, 4), dtype=w2c.dtype)
            w2c_4x4[:, :3, :4] = w2c
            w2c_4x4[:, 3, 3] = 1.0
            w2c = w2c_4x4
        c2w = np.linalg.inv(w2c).astype(np.float32)
    else:
        raise ValueError(f"Unexpected w2c shape: {w2c.shape}")
    
    return c2w


def scale_intrinsics(
    intrinsics: np.ndarray,
    from_width: int,
    from_height: int,
    to_width: int,
    to_height: int,
) -> np.ndarray:
    """
    Scale intrinsics from one image size to another.
    
    DepthAnything3 outputs intrinsics for processed_images size.
    We need to scale them to the original input image size for DepthSplat.
    
    Args:
        intrinsics: Intrinsics matrix [N, 3, 3] or [3, 3]
        from_width: Width of the image the intrinsics were computed for
        from_height: Height of the image the intrinsics were computed for
        to_width: Target width
        to_height: Target height
        
    Returns:
        Scaled intrinsics matrix [N, 3, 3] or [3, 3]
    """
    scale_x = to_width / from_width
    scale_y = to_height / from_height
    
    scaled = intrinsics.copy()
    
    if intrinsics.ndim == 2:
        # [3, 3]
        scaled[0, 0] *= scale_x  # fx
        scaled[0, 2] *= scale_x  # cx
        scaled[1, 1] *= scale_y  # fy
        scaled[1, 2] *= scale_y  # cy
    elif intrinsics.ndim == 3:
        # [N, 3, 3]
        scaled[:, 0, 0] *= scale_x  # fx
        scaled[:, 0, 2] *= scale_x  # cx
        scaled[:, 1, 1] *= scale_y  # fy
        scaled[:, 1, 2] *= scale_y  # cy
    else:
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    
    return scaled.astype(np.float32)


# ============================================================================
# Frame Synchronizer
# ============================================================================

class FrameSynchronizer:
    """
    Synchronizes frames from multiple camera streams for LOW LATENCY.
    
    This synchronizer is optimized for real-time streaming:
    - Only keeps the LATEST frame from each camera (plus minimal buffer for matching)
    - Always uses the NEWEST synchronized pair available
    - Discards older frames aggressively to minimize latency
    
    For test sources, uses frame count matching instead of timestamps
    since videotestsrc timestamps may not align.
    """
    
    def __init__(self, num_cameras: int = 2, max_time_diff_ms: float = 100.0, use_frame_count_sync: bool = False):
        self.num_cameras = num_cameras
        self.max_time_diff_ms = max_time_diff_ms
        self.use_frame_count_sync = use_frame_count_sync
        # For low latency: only keep 1 frame per camera (the latest)
        self.latest_frames = {i: None for i in range(num_cameras)}  # (key, timestamp_ns, frame)
        self.frame_counts = {i: 0 for i in range(num_cameras)}
        self.lock = threading.Lock()
        self.sequence_counter = 0
        
    def add_frame(self, camera_id: int, timestamp_ns: int, frame: np.ndarray) -> Optional[tuple]:
        """
        Add a frame from a camera and attempt to find a synchronized pair.
        
        LOW LATENCY: Always overwrites the previous frame from this camera.
        When a match is found, both latest frames are used and cleared.
        
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
            
            # LOW LATENCY: Always overwrite with the latest frame (drop old frames)
            self.latest_frames[camera_id] = (key, timestamp_ns, frame)
            
            # Try to find a matching pair using latest frames
            return self._try_match()
    
    def _try_match(self) -> Optional[tuple]:
        """Try to find a synchronized frame pair using the LATEST frames."""
        # Need frames from all cameras
        if any(self.latest_frames[i] is None for i in range(self.num_cameras)):
            return None
        
        frame_0_data = self.latest_frames[0]  # (key, timestamp_ns, frame)
        frame_1_data = self.latest_frames[1]
        
        key_0, ts_0, frame_0 = frame_0_data
        key_1, ts_1, frame_1 = frame_1_data
        
        if self.use_frame_count_sync:
            # Match by frame count - only match if counts are equal
            if key_0 != key_1:
                # Frames don't match - keep waiting
                # Discard the older one to stay fresh
                if key_0 < key_1:
                    self.latest_frames[0] = None  # Discard older frame from cam0
                else:
                    self.latest_frames[1] = None  # Discard older frame from cam1
                return None
            
            # Matching frame counts - use these frames
            self.latest_frames[0] = None
            self.latest_frames[1] = None
            
            seq_id = self.sequence_counter
            self.sequence_counter += 1
            
            return (seq_id, {
                0: (ts_0, frame_0),
                1: (ts_1, frame_1)
            })
        else:
            # Match by timestamp proximity
            diff_ms = abs(ts_0 - ts_1) / 1_000_000  # Convert ns to ms
            
            if diff_ms <= self.max_time_diff_ms:
                # Good match - use these frames
                self.latest_frames[0] = None
                self.latest_frames[1] = None
                
                seq_id = self.sequence_counter
                self.sequence_counter += 1
                
                return (seq_id, {
                    0: (ts_0, frame_0),
                    1: (ts_1, frame_1)
                })
            else:
                # Frames too far apart - discard the older one and wait
                if ts_0 < ts_1:
                    self.latest_frames[0] = None  # Discard older frame from cam0
                else:
                    self.latest_frames[1] = None  # Discard older frame from cam1
                return None


# ============================================================================
# GPU Worker Pool
# ============================================================================

class GPUWorker:
    """Worker that processes frame pairs on a specific GPU."""
    
    def __init__(self, gpu_id: int, worker_id: int, config: ServerConfig, on_available: Callable[['GPUWorker'], None] = None):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.config = config
        self.encoder: Optional[DepthSplatInference] = None
        self.depth_anything: Optional[DepthAnything3] = None
        # Latest task buffer (always process newest task for lowest latency)
        self.latest_task: Optional[tuple] = None
        self.latest_task_lock = threading.Lock()
        self.latest_task_event = threading.Event()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()  # Signals when models are loaded
        self._on_available = on_available  # Callback when worker becomes available
        self._is_processing = False  # Track if currently processing
        
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
        """Wait for the worker to be ready (models loaded)."""
        return self.ready_event.wait(timeout=timeout)
        
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.latest_task_event.set()  # Wake up the thread so it can exit
        if self.thread:
            self.thread.join(timeout=5.0)
            
    def submit(self, task: tuple):
        """Submit a task to this worker (overwrites any pending task)."""
        seq_id = task[0]
        with self.latest_task_lock:
            old_task = self.latest_task
            self.latest_task = task
            if old_task is not None:
                old_seq_id = old_task[0]
                # Call cancel callback for the replaced task
                if len(old_task) > 3 and old_task[3] is not None:
                    old_task[3](old_seq_id)  # cancel_callback(old_seq_id)
        self.latest_task_event.set()
        
    def _run(self):
        """Worker thread main loop."""
        # Initialize models on this GPU
        torch.cuda.set_device(self.gpu_id)
        device = torch.device(f"cuda:{self.gpu_id}")
        
        if self.config.verbose:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Initializing DepthAnything3 on GPU {self.gpu_id}")
        
        # Initialize DepthAnything3 for camera estimation
        self.depth_anything = DepthAnything3.from_pretrained(self.config.depth_anything_model)
        self.depth_anything = self.depth_anything.to(device=device)
        self.depth_anything.eval()
        
        if self.config.verbose:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Initializing DepthSplat encoder on GPU {self.gpu_id}")
        
        # Initialize DepthSplat encoder - explicitly pass device to ensure correct GPU
        self.encoder = DepthSplatInference(device=f"cuda:{self.gpu_id}")
        
        if self.config.verbose:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Both models ready")
        
        # Signal that we're ready
        self.ready_event.set()
        
        # Signal initial availability after models are loaded
        if self._on_available:
            self._on_available(self)
        
        while self.running:
            try:
                # Wait for a task to be available
                if not self.latest_task_event.wait(timeout=1.0):
                    continue  # Timeout, check if still running
                
                # Grab the latest task atomically
                with self.latest_task_lock:
                    if self.latest_task is None:
                        self.latest_task_event.clear()
                        continue
                    
                    task = self.latest_task
                    self.latest_task = None
                    self.latest_task_event.clear()
                    self._is_processing = True
                
                seq_id, frames_dict, result_callback, _cancel_callback = task
                self._process_task(seq_id, frames_dict, result_callback)
                
                # Mark as available and signal pool
                self._is_processing = False
                if self._on_available:
                    self._on_available(self)
                
            except Exception as e:
                print(f"[Worker {self.gpu_id}:{self.worker_id}] Error: {e}")
                self._is_processing = False
                if self._on_available:
                    self._on_available(self)
                
    def _process_task(self, seq_id: int, frames_dict: dict, result_callback: Callable):
        """Process a frame pair and invoke callback with result."""
        try:
            start_time = time.time()
            
            # Extract frames - HWC uint8 numpy arrays
            frame_0 = frames_dict[0][1]  # (timestamp, frame) -> frame
            frame_1 = frames_dict[1][1]
            
            # Get original image dimensions
            orig_height, orig_width = frame_0.shape[:2]
            
            # Save camera input frames if enabled
            if self.config.camera_input_dir:
                save_dir = Path(self.config.camera_input_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save both frames as PNG
                img_0 = Image.fromarray(frame_0)
                img_1 = Image.fromarray(frame_1)
                
                path_0 = save_dir / f"seq_{seq_id:05d}_cam0.png"
                path_1 = save_dir / f"seq_{seq_id:05d}_cam1.png"
                
                img_0.save(path_0)
                img_1.save(path_1)
                
                if self.config.verbose:
                    print(f"[Worker {self.gpu_id}:{self.worker_id}] Saved input frames to {path_0} and {path_1}")
            
            
            # Run DepthAnything3 to get intrinsics and extrinsics (W2C)
            # Pass numpy arrays directly - DepthAnything3 accepts HWC uint8 arrays
            da3_start = time.time()
            prediction = self.depth_anything.inference(
                image=[frame_0, frame_1],  # List of HWC uint8 numpy arrays
                process_res=self.config.depth_anything_process_res,
                process_res_method="upper_bound_resize",
            )
            da3_elapsed = time.time() - da3_start
            
            
            # Get processed image size from DepthAnything3 for intrinsics scaling
            processed_height, processed_width = prediction.processed_images.shape[1:3]
            
            # DepthAnything3 outputs W2C extrinsics - convert to C2W poses for DepthSplat
            # DepthSplat's "extrinsics" parameter actually expects C2W (camera-to-world) poses
            w2c_extrinsics = prediction.extrinsics  # [N, 3, 4] or [N, 4, 4] - W2C
            c2w_poses = w2c_to_c2w(w2c_extrinsics)  # [N, 4, 4] - C2W
            
            # Scale intrinsics from processed_images size to original input size
            da3_intrinsics = prediction.intrinsics  # [N, 3, 3] for processed size
            scaled_intrinsics = scale_intrinsics(
                da3_intrinsics,
                from_width=processed_width,
                from_height=processed_height,
                to_width=orig_width,
                to_height=orig_height,
            )
            
            
            # Convert frames to tensors for DepthSplat - use ORIGINAL images, not processed
            # Convert from HWC uint8 to CHW float [0, 1]
            frame_0_tensor = torch.from_numpy(frame_0).permute(2, 0, 1).float() / 255.0
            frame_1_tensor = torch.from_numpy(frame_1).permute(2, 0, 1).float() / 255.0
            images = torch.stack([frame_0_tensor, frame_1_tensor], dim=0)
            
            # Prepare lists for DepthSplat
            intrinsics_list = [scaled_intrinsics[0], scaled_intrinsics[1]]
            extrinsics_list = [c2w_poses[0], c2w_poses[1]]  # C2W poses
            original_sizes = [
                (orig_width, orig_height),
                (orig_width, orig_height),
            ]
            
            
            # Run DepthSplat encoder inference
            encoder_start = time.time()
            inference_config = InferenceConfig(
                target_height=self.config.target_height,
                target_width=self.config.target_width,
                near_disparity=self.config.near_disparity,
                far_disparity=self.config.far_disparity,
                verbose=False,  # Suppress verbose output in streaming mode
                output_format=self.config.output_format,
            )
            
            output_bytes = self.encoder.run_from_data(
                images=images,
                intrinsics_list=intrinsics_list,
                extrinsics_list=extrinsics_list,
                original_sizes=original_sizes,
                image_filenames=[f"cam0_seq{seq_id}", f"cam1_seq{seq_id}"],
                config=inference_config,
            )
            encoder_elapsed = time.time() - encoder_start
            
            total_elapsed = time.time() - start_time
            
            if self.config.verbose:
                print(f"[Worker {self.gpu_id}:{self.worker_id}] Seq {seq_id} completed in {total_elapsed:.2f}s")
            
            # Invoke callback with result
            result_callback(seq_id, output_bytes)
            
        except Exception as e:
            print(f"[Worker {self.gpu_id}:{self.worker_id}] Failed to process seq {seq_id}: {e}")
            import traceback
            traceback.print_exc()
            # Notify sequencer of failure so it doesn't block on this seq
            # Pass None as output_bytes to indicate failure
            result_callback(seq_id, None)


class GPUWorkerPool:
    """Pool of GPU workers for distributed inference."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.workers: list[GPUWorker] = []
        self.worker_index = 0
        self.lock = threading.Lock()
        
        # Available worker tracking for pull-based model
        self._available_workers: list[GPUWorker] = []
        self._available_lock = threading.Lock()
        self._available_event = threading.Event()  # Signals when a worker becomes available
        
    def _on_worker_available(self, worker: GPUWorker):
        """Called when a worker becomes available for new work."""
        with self._available_lock:
            if worker not in self._available_workers:
                self._available_workers.append(worker)
        self._available_event.set()
        
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
                worker = GPUWorker(gpu_id, worker_idx, self.config, on_available=self._on_worker_available)
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
    
    def wait_for_available_worker(self, timeout: float = None) -> Optional[GPUWorker]:
        """
        Wait for a worker to become available (pull-based model).
        
        This blocks until a worker finishes processing and is ready for new work.
        Use this for lowest-latency streaming where you want to process only
        the newest frame when a worker becomes free.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            An available GPUWorker, or None if timeout occurred
        """
        start_time = time.time()
        
        while True:
            # Check for available workers
            with self._available_lock:
                if self._available_workers:
                    worker = self._available_workers.pop(0)
                    # Clear event if no more workers available
                    if not self._available_workers:
                        self._available_event.clear()
                    return worker
            
            # Calculate remaining timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    return None
            else:
                remaining = 1.0  # Use 1s polling interval when no timeout
            
            # Wait for availability signal
            self._available_event.wait(timeout=min(remaining, 1.0))
    
    def submit_to_worker(self, worker: GPUWorker, seq_id: int, frames_dict: dict, 
                         result_callback: Callable, cancel_callback: Callable = None):
        """Submit a task to a specific worker (for pull-based model)."""
        with self._available_lock:
            # Remove from available list if present (shouldn't be, but just in case)
            if worker in self._available_workers:
                self._available_workers.remove(worker)
        worker.submit((seq_id, frames_dict, result_callback, cancel_callback))
    
    def get_in_flight_count(self) -> int:
        """Get number of frames currently being processed or waiting."""
        with self._available_lock:
            available_count = len(self._available_workers)
        return len(self.workers) - available_count
        
    def submit(self, seq_id: int, frames_dict: dict, result_callback: Callable, cancel_callback: Callable = None):
        """Submit a task to the next available worker (round-robin - legacy push model)."""
        with self.lock:
            worker = self.workers[self.worker_index]
            self.worker_index = (self.worker_index + 1) % len(self.workers)
        
        worker.submit((seq_id, frames_dict, result_callback, cancel_callback))


# ============================================================================
# Output Sequencer
# ============================================================================

class OutputSequencer:
    """
    Emits PLY outputs in submission order.
    
    Tracks which seq_ids are in-flight (submitted to workers) and emits
    results in the order they were submitted. This ensures correct ordering
    with multiple GPUs while still allowing frame skipping.
    """
    
    def __init__(
        self,
        config: ServerConfig,
        on_complete: Optional[Callable[[int], None]] = None,
        save_files: bool = True,
    ):
        """
        Initialize output sequencer.
        
        Args:
            config: Server configuration
            on_complete: Callback when a frame is emitted (receives total count)
            save_files: Whether to save files to disk (can be False for streaming-only)
        """
        self.config = config
        self.total_emitted = 0
        self.lock = threading.Lock()
        self.save_files = save_files
        self.output_dir = Path(config.output_dir)
        if self.save_files:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_complete = on_complete  # Callback when a PLY is emitted
        
        # Track in-flight seq_ids in submission order
        self.in_flight: list[int] = []  # Ordered list of seq_ids submitted to workers
        self.results_buffer: dict[int, bytes] = {}  # Buffer for completed but not-yet-emitted results
        
    def mark_submitted(self, seq_id: int):
        """
        Mark a seq_id as submitted to a worker.
        Must be called before add_result for correct ordering.
        """
        with self.lock:
            self.in_flight.append(seq_id)
    
    def cancel_submitted(self, seq_id: int):
        """
        Remove a seq_id from in-flight when it's replaced/cancelled.
        Called by workers when a pending task is replaced by a newer one.
        """
        with self.lock:
            if seq_id in self.in_flight:
                self.in_flight.remove(seq_id)
        
    def add_result(self, seq_id: int, output_bytes: bytes):
        """
        Add a result and emit if it's the next in submission order.
        
        Args:
            seq_id: Sequence ID of the result
            output_bytes: Output file contents (SPZ or PLY)
        """
        with self.lock:
            # Buffer this result
            self.results_buffer[seq_id] = output_bytes
            
            # Emit all results that are ready (in submission order)
            self._emit_ready()
            
    def _emit_ready(self):
        """Emit results that are ready in submission order."""
        while self.in_flight and self.in_flight[0] in self.results_buffer:
            seq_id = self.in_flight.pop(0)
            output_bytes = self.results_buffer.pop(seq_id)
            
            # Handle failed sequences (None output_bytes)
            if output_bytes is None:
                print(f"[Output] Seq {seq_id} failed, skipping")
                continue
            
            size_mb = len(output_bytes) / (1024 * 1024)
            
            # Emit via callback (e.g., WebSocket streaming)
            if self.config.output_callback:
                try:
                    self.config.output_callback(seq_id, output_bytes)
                    if self.config.verbose:
                        print(f"[Output] Sent seq {seq_id} via callback ({size_mb:.2f} MB)")
                except Exception as e:
                    print(f"Output callback error for seq {seq_id}: {e}")
            
            # Save to file if enabled
            if self.save_files:
                ext = self.config.output_format.lower()
                output_path = self.output_dir / f"output_{self.total_emitted:06d}.{ext}"
                with open(output_path, 'wb') as f:
                    f.write(output_bytes)
                
                if self.config.verbose:
                    print(f"[Output] Saved seq {seq_id} to {output_path} ({size_mb:.2f} MB)")
            
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
        
        # Note: videorate removed because it blocks when input has no timestamps (pts=none)
        # The pipeline still works without it - we just accept whatever framerate comes in
        pipeline = (
            f"{self.source_pipeline} ! "
            f"videoscale method=4 ! "  # method=4 is Lanczos (highest quality)
            f"video/x-raw,width={width},height={height} ! "
            f"videoconvert ! "
            f"video/x-raw,format=RGB ! "
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )
        return pipeline
        
    def start(self):
        """Start the camera pipeline."""
        pipeline_str = self.build_pipeline()
        
        if self.config.verbose:
            print(f"[Camera {self.camera_id}] Pipeline: {pipeline_str}")
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Connect bus to catch errors and state changes
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::warning", self._on_bus_warning)
        bus.connect("message::state-changed", self._on_state_changed)
        bus.connect("message::eos", self._on_eos)
        # Connect appsink signal
        appsink = self.pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_new_sample)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            # Try to get more error details from the bus
            bus = self.pipeline.get_bus()
            if bus:
                msg = bus.pop_filtered(Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    raise RuntimeError(
                        f"Failed to start camera {self.camera_id} pipeline: {err.message}\n"
                        f"Debug info: {debug}"
                    )
            raise RuntimeError(f"Failed to start camera {self.camera_id} pipeline")
        elif ret == Gst.StateChangeReturn.ASYNC:
            # For sources like tcpserversrc, the state change is async (waiting for connection)
            if self.config.verbose:
                print(f"[Camera {self.camera_id}] Waiting for connection...")
        
        if self.config.verbose:
            print(f"[Camera {self.camera_id}] Started")
    
    def _on_bus_error(self, bus, message):
        """Handle GStreamer error messages."""
        err, debug = message.parse_error()
        print(f"[Camera {self.camera_id}] ERROR: {err.message}")
        if debug:
            print(f"[Camera {self.camera_id}] Debug: {debug}")
    
    def _on_bus_warning(self, bus, message):
        """Handle GStreamer warning messages."""
        warn, debug = message.parse_warning()
        print(f"[Camera {self.camera_id}] WARNING: {warn.message}")
    
    def _on_state_changed(self, bus, message):
        """Handle pipeline state changes."""
        if message.src == self.pipeline:
            old, new, pending = message.parse_state_changed()
            if self.config.verbose:
                print(f"[Camera {self.camera_id}] State: {old.value_nick} -> {new.value_nick}")
    
    def _on_eos(self, bus, message):
        """Handle end of stream."""
        print(f"[Camera {self.camera_id}] End of stream")
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state as string."""
        if self.pipeline:
            ret, state, pending = self.pipeline.get_state(0)
            return f"{state.value_nick}" + (f" (pending: {pending.value_nick})" if pending != Gst.State.VOID_PENDING else "")
        return "no pipeline"
            
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
        # For timestamp sync, always use wall-clock time for better cross-stream sync
        # PTS timestamps are stream-relative and won't match between cameras that start at different times
        if self.config.use_frame_count_sync:
            timestamp_ns = buffer.pts if buffer.pts != Gst.CLOCK_TIME_NONE else time.time_ns()
        else:
            timestamp_ns = time.time_ns()
        
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
# WebRTC Camera Pipeline
# ============================================================================

class WebRTCCameraPipeline:
    """
    WebRTC-based camera pipeline for receiving streams from browsers/mobile apps.
    
    Uses GStreamer's webrtcbin element to handle WebRTC connections.
    Each instance handles one camera stream (camera_id 0 or 1).
    """
    
    def __init__(
        self,
        camera_id: int,
        config: ServerConfig,
        frame_callback: Callable[[int, int, np.ndarray], None],
    ):
        """
        Initialize WebRTC camera pipeline.
        
        Args:
            camera_id: Camera index (0 or 1)
            config: Server configuration
            frame_callback: Callback for received frames (camera_id, timestamp_ns, frame)
        """
        self.camera_id = camera_id
        self.config = config
        self.frame_callback = frame_callback
        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtcbin: Optional[Gst.Element] = None
        self.frame_count = 0
        self.raw_frame_count = 0  # Counter for raw frames (before scaling)
        self.websocket = None  # Will be set when client connects
        self.connected = False  # True when WebRTC connection is established
        self.lock = threading.Lock()
        self.signaling_loop = None  # Event loop for signaling (set by signaling server)
        
    def start(self):
        """Initialize the WebRTC pipeline (waits for incoming connection)."""
        # Create pipeline with webrtcbin
        self.pipeline = Gst.Pipeline.new(f"webrtc-cam-{self.camera_id}")
        if not self.pipeline:
            raise RuntimeError(f"Failed to create pipeline for camera {self.camera_id}")
        
        # Create webrtcbin element
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", f"webrtcbin-{self.camera_id}")
        if not self.webrtcbin:
            raise RuntimeError("Failed to create webrtcbin element. Is gst-plugins-bad installed?")
        
        # Configure webrtcbin for receiving streams
        self.webrtcbin.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self.webrtcbin.set_property("stun-server", self.config.stun_server)
        
        # Connect signals
        self.webrtcbin.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtcbin.connect("pad-added", self._on_pad_added)
        
        self.pipeline.add(self.webrtcbin)
        
        # Set pipeline to PLAYING state immediately - webrtcbin needs to be playing
        # to handle incoming offers and create answers
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print(f"[WebRTC Camera {self.camera_id}] ERROR: Failed to set pipeline to PLAYING state")
            # Get the bus and check for error messages
            bus = self.pipeline.get_bus()
            if bus:
                msg = bus.pop_filtered(Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    print(f"[WebRTC Camera {self.camera_id}] GStreamer error: {err.message}")
                    print(f"[WebRTC Camera {self.camera_id}] Debug info: {debug}")
        elif ret == Gst.StateChangeReturn.ASYNC:
            # Wait for state change to complete
            self.pipeline.get_state(5 * Gst.SECOND)
        
        if self.config.verbose:
            print(f"[WebRTC Camera {self.camera_id}] Pipeline ready")
            
    def stop(self):
        """Stop the WebRTC pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            if self.config.verbose:
                print(f"[WebRTC Camera {self.camera_id}] Stopped (received {self.frame_count} frames)")
    
    def set_websocket(self, websocket):
        """Set the WebSocket connection for signaling."""
        with self.lock:
            self.websocket = websocket
            if websocket is None:
                self.connected = False
                # Don't change pipeline state - it needs to stay PLAYING for new connections
    
    def handle_sdp_offer(self, sdp_text: str):
        """Handle incoming SDP offer from client."""
        # Schedule on GLib main loop to ensure thread safety with GStreamer
        GLib.idle_add(self._handle_sdp_offer_impl, sdp_text)
    
    def _handle_sdp_offer_impl(self, sdp_text: str):
        """Implementation of SDP offer handling (runs on GLib main loop)."""
        if self.config.verbose:
            print(f"[WebRTC Camera {self.camera_id}] Received SDP offer")
        
        # Mark as connected (WebRTC negotiation started)
        self.connected = True
        
        # Parse SDP
        res, sdp_msg = GstSdp.SDPMessage.new()
        if res != GstSdp.SDPResult.OK:
            print(f"[WebRTC Camera {self.camera_id}] Failed to create SDP message")
            return False
        
        res = GstSdp.sdp_message_parse_buffer(sdp_text.encode(), sdp_msg)
        if res != GstSdp.SDPResult.OK:
            print(f"[WebRTC Camera {self.camera_id}] Failed to parse SDP offer")
            return False
        
        
        # Create offer object
        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdp_msg
        )
        
        # Set remote description
        promise = Gst.Promise.new_with_change_func(self._on_offer_set, None)
        self.webrtcbin.emit("set-remote-description", offer, promise)
        
        return False  # Don't repeat
    
    def handle_ice_candidate(self, candidate: str, sdp_mline_index: int):
        """Handle incoming ICE candidate from client."""
        # Schedule on GLib main loop to ensure thread safety with GStreamer
        GLib.idle_add(self._handle_ice_candidate_impl, candidate, sdp_mline_index)
    
    def _handle_ice_candidate_impl(self, candidate: str, sdp_mline_index: int):
        """Implementation of ICE candidate handling (runs on GLib main loop)."""
        self.webrtcbin.emit("add-ice-candidate", sdp_mline_index, candidate)
        return False  # Don't repeat
    
    def _on_offer_set(self, promise, user_data):
        """Callback when remote description is set."""
        promise.wait()
        
        # Create answer
        promise = Gst.Promise.new_with_change_func(self._on_answer_created, None)
        self.webrtcbin.emit("create-answer", None, promise)
    
    def _on_answer_created(self, promise, user_data):
        """Callback when answer is created."""
        promise.wait()
        
        state = promise.get_reply()
        if state is None:
            print(f"[WebRTC Camera {self.camera_id}] Failed to create answer")
            return
        
        answer = state.get_value("answer")
        if answer is None:
            error = state.get_value("error")
            if error:
                print(f"[WebRTC Camera {self.camera_id}] Error creating answer: {error}")
            return
        
        # Set local description
        promise = Gst.Promise.new_with_change_func(self._on_local_description_set, None)
        self.webrtcbin.emit("set-local-description", answer, promise)
    
    def _on_local_description_set(self, promise, user_data):
        """Callback when local description is set."""
        promise.wait()
        
        # Get the local description from webrtcbin
        local_desc = self.webrtcbin.get_property("local-description")
        if local_desc is None:
            print(f"[WebRTC Camera {self.camera_id}] ERROR: No local description available")
            return
        
        # Send answer to client via WebSocket
        sdp_text = local_desc.sdp.as_text()
        self._send_signaling_message({
            "type": "answer",
            "sdp": sdp_text,
        })
    
    def _on_negotiation_needed(self, webrtcbin):
        """Called when negotiation is needed."""
        pass  # We receive offers from clients, not initiate them
    
    def _on_ice_candidate(self, webrtcbin, sdp_mline_index: int, candidate: str):
        """Called when we have a new ICE candidate to send to the peer."""
        self._send_signaling_message({
            "type": "ice",
            "candidate": candidate,
            "sdpMLineIndex": sdp_mline_index,
        })
    
    def _on_pad_added(self, webrtcbin, pad: Gst.Pad):
        """Called when a new pad is added (incoming stream)."""
        if pad.direction != Gst.PadDirection.SRC:
            return
        
        # Get the pad caps to determine media type
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)
        
        if caps is None or caps.is_empty():
            print(f"[WebRTC Camera {self.camera_id}] No caps on pad")
            return
        
        struct = caps.get_structure(0)
        media_type = struct.get_name()
        
        # Only handle video streams
        if not media_type.startswith("application/x-rtp"):
            return
        
        # Check if it's video
        encoding_name = struct.get_string("encoding-name")
        media = struct.get_string("media")
        
        if media != "video":
            return
        
        # Create decoding pipeline for the incoming video
        self._setup_decoding_pipeline(pad, encoding_name)
    
    def _setup_decoding_pipeline(self, src_pad: Gst.Pad, encoding_name: str):
        """Set up decoding pipeline for incoming video."""
        # Use decodebin for automatic codec selection
        decodebin = Gst.ElementFactory.make("decodebin", f"decode-{self.camera_id}")
        if not decodebin:
            print(f"[WebRTC Camera {self.camera_id}] Failed to create decodebin")
            return
        
        decodebin.connect("pad-added", self._on_decodebin_pad)
        
        # Create appropriate depayloader based on encoding
        depayloader = None
        if encoding_name.upper() == "H264":
            depayloader = Gst.ElementFactory.make("rtph264depay", f"depay-{self.camera_id}")
        elif encoding_name.upper() == "VP8":
            depayloader = Gst.ElementFactory.make("rtpvp8depay", f"depay-{self.camera_id}")
        elif encoding_name.upper() == "VP9":
            depayloader = Gst.ElementFactory.make("rtpvp9depay", f"depay-{self.camera_id}")
        else:
            # Try generic depayloader via decodebin
            if self.config.verbose:
                print(f"[WebRTC Camera {self.camera_id}] Unknown encoding {encoding_name}, using decodebin")
        
        if depayloader:
            self.pipeline.add(depayloader)
            self.pipeline.add(decodebin)
            
            depayloader.sync_state_with_parent()
            decodebin.sync_state_with_parent()
            
            # Link: webrtcbin pad -> depayloader -> decodebin
            src_pad.link(depayloader.get_static_pad("sink"))
            depayloader.link(decodebin)
        else:
            # Direct to decodebin (it will handle depayloading)
            self.pipeline.add(decodebin)
            decodebin.sync_state_with_parent()
            src_pad.link(decodebin.get_static_pad("sink"))
    
    def _on_decodebin_pad(self, decodebin, pad: Gst.Pad):
        """Called when decodebin creates a new pad (decoded stream)."""
        caps = pad.get_current_caps()
        if caps is None:
            return
        
        struct = caps.get_structure(0)
        name = struct.get_name()
        
        if not name.startswith("video/"):
            return
        
        # Create conversion and appsink pipeline
        videoconvert = Gst.ElementFactory.make("videoconvert", f"convert-{self.camera_id}")
        videoscale = Gst.ElementFactory.make("videoscale", f"scale-{self.camera_id}")
        
        # Use Lanczos scaling (method=4) for highest quality
        # Methods: 0=nearest, 1=bilinear, 2=4-tap, 3=9-tap, 4=Lanczos
        videoscale.set_property("method", 4)
        
        capsfilter = Gst.ElementFactory.make("capsfilter", f"caps-{self.camera_id}")
        appsink = Gst.ElementFactory.make("appsink", f"sink-{self.camera_id}")
        
        if not all([videoconvert, videoscale, capsfilter, appsink]):
            print(f"[WebRTC Camera {self.camera_id}] Failed to create conversion elements")
            return
        
        # Configure capsfilter for RGB output at target resolution
        width = self.config.input_width
        height = self.config.input_height
        out_caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={width},height={height}")
        capsfilter.set_property("caps", out_caps)
        
        # Configure appsink
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 2)
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_new_sample)
        
        # Add elements to pipeline
        self.pipeline.add(videoconvert)
        self.pipeline.add(videoscale)
        self.pipeline.add(capsfilter)
        self.pipeline.add(appsink)
        
        # If saving raw camera input, add a tee after videoconvert to capture pre-scaling frames
        if self.config.camera_input_dir:
            tee = Gst.ElementFactory.make("tee", f"tee-{self.camera_id}")
            queue_main = Gst.ElementFactory.make("queue", f"queue-main-{self.camera_id}")
            queue_raw = Gst.ElementFactory.make("queue", f"queue-raw-{self.camera_id}")
            raw_capsfilter = Gst.ElementFactory.make("capsfilter", f"raw-caps-{self.camera_id}")
            raw_appsink = Gst.ElementFactory.make("appsink", f"raw-sink-{self.camera_id}")
            
            if not all([tee, queue_main, queue_raw, raw_capsfilter, raw_appsink]):
                print(f"[WebRTC Camera {self.camera_id}] Failed to create raw capture elements, continuing without raw capture")
            else:
                # Configure raw capsfilter - just convert to RGB, no scaling
                raw_capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
                
                # Configure raw appsink
                raw_appsink.set_property("emit-signals", True)
                raw_appsink.set_property("max-buffers", 2)
                raw_appsink.set_property("drop", True)
                raw_appsink.connect("new-sample", self._on_raw_sample)
                
                # Add elements
                self.pipeline.add(tee)
                self.pipeline.add(queue_main)
                self.pipeline.add(queue_raw)
                self.pipeline.add(raw_capsfilter)
                self.pipeline.add(raw_appsink)
                
                # Sync states for all elements
                videoconvert.sync_state_with_parent()
                tee.sync_state_with_parent()
                queue_main.sync_state_with_parent()
                queue_raw.sync_state_with_parent()
                videoscale.sync_state_with_parent()
                capsfilter.sync_state_with_parent()
                appsink.sync_state_with_parent()
                raw_capsfilter.sync_state_with_parent()
                raw_appsink.sync_state_with_parent()
                
                # Link: decodebin -> videoconvert -> tee
                #                                     ├-> queue_main -> videoscale -> capsfilter -> appsink (scaled)
                #                                     └-> queue_raw -> raw_capsfilter -> raw_appsink (raw)
                pad.link(videoconvert.get_static_pad("sink"))
                videoconvert.link(tee)
                
                # Main branch (scaled)
                tee.link(queue_main)
                queue_main.link(videoscale)
                videoscale.link(capsfilter)
                capsfilter.link(appsink)
                
                # Raw branch (unscaled)
                tee.link(queue_raw)
                queue_raw.link(raw_capsfilter)
                raw_capsfilter.link(raw_appsink)
                
                print(f"[WebRTC Camera {self.camera_id}] Video stream connected (with raw capture)")
                return
        
        # Standard pipeline without raw capture
        # Sync states
        videoconvert.sync_state_with_parent()
        videoscale.sync_state_with_parent()
        capsfilter.sync_state_with_parent()
        appsink.sync_state_with_parent()
        
        # Link: decodebin pad -> videoconvert -> videoscale -> capsfilter -> appsink
        pad.link(videoconvert.get_static_pad("sink"))
        videoconvert.link(videoscale)
        videoscale.link(capsfilter)
        capsfilter.link(appsink)
        
        print(f"[WebRTC Camera {self.camera_id}] Video stream connected")
    
    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new decoded video frame."""
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
            ).copy()
            
            # Invoke callback
            self.frame_callback(self.camera_id, timestamp_ns, frame)
            
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _on_raw_sample(self, appsink) -> Gst.FlowReturn:
        """Handle raw video frame (before scaling) for debugging."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        
        # Get buffer
        buffer = sample.get_buffer()
        
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
            ).copy()
            
            # Save raw frame immediately
            self.raw_frame_count += 1
            save_dir = Path(self.config.camera_input_dir) / "raw_webrtc"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            img = Image.fromarray(frame)
            save_path = save_dir / f"cam{self.camera_id}_raw_{self.raw_frame_count:06d}_{width}x{height}.png"
            img.save(save_path)
            
            if self.config.verbose and self.raw_frame_count <= 5:
                print(f"[WebRTC Camera {self.camera_id}] Saved raw frame {self.raw_frame_count}: {width}x{height} -> {save_path}")
            
        finally:
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _send_signaling_message(self, message: dict):
        """Send a signaling message via WebSocket."""
        with self.lock:
            if self.websocket is None:
                print(f"[WebRTC Camera {self.camera_id}] No WebSocket connection for signaling")
                return
            
            if self.signaling_loop is None:
                print(f"[WebRTC Camera {self.camera_id}] No signaling event loop available")
                return
            
            # Schedule send on the signaling event loop
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps(message)),
                    self.signaling_loop
                )
            except Exception as e:
                print(f"[WebRTC Camera {self.camera_id}] Failed to send signaling message: {e}")


class WebRTCSignalingServer:
    """
    WebSocket-based signaling server for WebRTC connections.
    
    Handles SDP offer/answer exchange and ICE candidate relay between
    browser/mobile clients and the WebRTC camera pipelines.
    """
    
    def __init__(
        self,
        config: ServerConfig,
        camera_pipelines: Dict[int, WebRTCCameraPipeline],
    ):
        """
        Initialize the signaling server.
        
        Args:
            config: Server configuration
            camera_pipelines: Dict mapping camera_id to WebRTCCameraPipeline
        """
        self.config = config
        self.camera_pipelines = camera_pipelines
        self.server = None
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        
    def start(self):
        """Start the signaling server in a background thread."""
        self.running = True
        self.thread = threading.Thread(
            target=self._run_server,
            name="WebRTCSignaling",
            daemon=True
        )
        self.thread.start()
        
        # Wait for server to start
        time.sleep(1.0)
        
    def stop(self):
        """Stop the signaling server."""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _run_server(self):
        """Run the async server in this thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._start_server())
            self.loop.run_forever()
        finally:
            self.loop.close()
    
    async def _start_server(self):
        """Start the WebSocket server."""
        try:
            import websockets
            from websockets.server import serve
        except ImportError:
            print("ERROR: websockets package not installed. Install with: pip install websockets")
            return
        
        host = "0.0.0.0"
        port = self.config.webrtc_port
        
        # Set up SSL context if certificates are provided
        ssl_context = None
        protocol = "ws"
        if self.config.ssl_cert and self.config.ssl_key:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)
            protocol = "wss"
        
        
        self.server = await serve(
            self._handle_client,
            host,
            port,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=20,
        )
        
        print(f"[WebRTC Signaling] Server ready at {protocol}://{host}:{port}")
    
    async def _handle_client(self, websocket, path=None):
        """Handle a WebSocket client connection."""
        camera_id = None
        pipeline = None
        
        if self.config.verbose:
            print(f"[WebRTC Signaling] Client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "register":
                        # Client registers for a specific camera
                        camera_id = data.get("camera_id", 0)
                        
                        if camera_id not in self.camera_pipelines:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": f"Invalid camera_id: {camera_id}. Valid: {list(self.camera_pipelines.keys())}"
                            }))
                            continue
                        
                        pipeline = self.camera_pipelines[camera_id]
                        pipeline.set_websocket(websocket)
                        pipeline.signaling_loop = self.loop  # Give pipeline access to our event loop
                        
                        print(f"[WebRTC] Camera {camera_id} client connected")
                        
                        await websocket.send(json.dumps({
                            "type": "registered",
                            "camera_id": camera_id,
                        }))
                    
                    elif msg_type == "offer":
                        # SDP offer from client
                        if pipeline is None:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": "Must register before sending offer"
                            }))
                            continue
                        
                        sdp = data.get("sdp", "")
                        pipeline.handle_sdp_offer(sdp)
                    
                    elif msg_type == "ice":
                        # ICE candidate from client
                        if pipeline is None:
                            continue
                        
                        candidate = data.get("candidate", "")
                        sdp_mline_index = data.get("sdpMLineIndex", 0)
                        
                        if candidate:  # Ignore empty candidates (end-of-candidates)
                            pipeline.handle_ice_candidate(candidate, sdp_mline_index)
                    
                    else:
                        if self.config.verbose:
                            print(f"[WebRTC Signaling] Unknown message type: {msg_type}")
                
                except json.JSONDecodeError as e:
                    print(f"[WebRTC Signaling] Invalid JSON: {e}")
                except Exception as e:
                    print(f"[WebRTC Signaling] Error handling message: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            if self.config.verbose:
                print(f"[WebRTC] Client disconnected: {e}")
        finally:
            if pipeline:
                pipeline.set_websocket(None)
                print(f"[WebRTC] Camera {camera_id} client disconnected")


class WebClientServer:
    """
    Simple HTTP server to serve the WebRTC client HTML page.
    
    This makes it easy to access the client from any device on the network.
    Note: This serves over HTTP only. The WebSocket signaling server handles SSL separately.
    
    Supports RunPod environment variables for auto-configuration:
    - RUNPOD_PUBLIC_IP: Pre-fills the server host field
    - RUNPOD_TCP_PORT_8080: Pre-fills the signaling port field
    """
    
    def __init__(self, port: int = 8888, verbose: bool = True, signaling_port: int = 8080):
        """
        Initialize the web client server.
        
        Args:
            port: Port to serve on (default: 8888)
            verbose: Enable verbose logging
            signaling_port: WebRTC signaling port (for default value in HTML)
        """
        self.port = port
        self.verbose = verbose
        self.signaling_port = signaling_port
        self.thread: Optional[threading.Thread] = None
        self.server = None
        self.running = False
        
        # Find the HTML file
        self.html_file = Path(__file__).parent / "webrtc_client_example.html"
        if not self.html_file.exists():
            print(f"[WebClient] Warning: Client HTML not found at {self.html_file}")
        
        # Check for RunPod environment variables
        import os
        self.runpod_ip = os.environ.get("RUNPOD_PUBLIC_IP", "")
        self.runpod_port = os.environ.get("RUNPOD_TCP_PORT_8080", "")
    
    def start(self):
        """Start the HTTP server in a background thread."""
        if not self.html_file.exists():
            print(f"[WebClient] Cannot start - HTML file not found")
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._run_server,
            name="WebClientServer",
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop the HTTP server."""
        self.running = False
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _run_server(self):
        """Run the HTTP server."""
        import http.server
        import socketserver
        
        html_dir = self.html_file.parent
        html_name = self.html_file.name
        runpod_ip = self.runpod_ip
        runpod_port = self.runpod_port
        signaling_port = self.signaling_port
        verbose = self.verbose
        
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, directory=None, **kwargs):
                super().__init__(*args, directory=str(html_dir), **kwargs)
            
            def log_message(self, format, *args):
                # Suppress default logging unless verbose
                pass
            
            def do_GET(self):
                # Serve the HTML file for root path with templated values
                if self.path == '/' or self.path == '/index.html' or self.path == '/' + html_name:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # Read and template the HTML
                    html_path = html_dir / html_name
                    with open(html_path, 'r') as f:
                        html_content = f.read()
                    
                    # Inject RunPod environment variables as JavaScript
                    # This allows the client to use these values for auto-configuration
                    inject_script = f'''<script>
        // Auto-configuration from server environment
        window.RUNPOD_PUBLIC_IP = "{runpod_ip}";
        window.RUNPOD_TCP_PORT_8080 = "{runpod_port}";
        window.DEFAULT_SIGNALING_PORT = "{signaling_port}";
    </script>
</head>'''
                    html_content = html_content.replace('</head>', inject_script)
                    
                    self.wfile.write(html_content.encode('utf-8'))
                    return
                return super().do_GET()
        
        try:
            server = socketserver.TCPServer(("0.0.0.0", self.port), CustomHandler)
            self.server = server
            if verbose:
                print(f"[WebClient] Serving client at http://0.0.0.0:{self.port}")
                if runpod_ip:
                    print(f"[WebClient] RunPod IP detected: {runpod_ip}")
                if runpod_port:
                    print(f"[WebClient] RunPod port detected: {runpod_port}")
            server.serve_forever()
        except OSError as e:
            print(f"[WebClient] Failed to start server on port {self.port}: {e}")
        except Exception as e:
            print(f"[WebClient] Server error: {e}")
            import traceback
            traceback.print_exc()


class WebRTCDepthSplatServer:
    """
    DepthSplat server using WebRTC for camera input.
    
    Similar to DepthSplatServer but uses WebRTC camera pipelines
    instead of traditional GStreamer sources.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or CONFIG
        self.synchronizer = FrameSynchronizer(
            num_cameras=2,
            use_frame_count_sync=True  # WebRTC streams use frame count sync
        )
        self.worker_pool: Optional[GPUWorkerPool] = None
        self.output_sequencer: Optional[OutputSequencer] = None
        self.cameras: Dict[int, WebRTCCameraPipeline] = {}
        self.signaling_server: Optional[WebRTCSignalingServer] = None
        self.web_client_server: Optional[WebClientServer] = None
        self.main_loop: Optional[GLib.MainLoop] = None
        self.running = False
        
        # Rate limiting
        self.last_process_time = 0.0
        self.min_frame_interval = 1.0 / self.config.max_fps if self.config.max_fps > 0 else 0.0
        
        # Latest frame buffer (always process newest frame for lowest latency)
        self.latest_frame: Optional[tuple] = None  # Holds (seq_id, frames_dict) or None
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_event = threading.Event()  # Signals when new frame available
        self.process_thread: Optional[threading.Thread] = None
        
        # Counters
        self.start_time: float = 0.0
        self.frame_pairs_queued = 0
        self.frame_pairs_processed = 0
        self.processing_seq_counter = 0
        self.seq_counter_lock = threading.Lock()
    
    def start(self):
        """Start the WebRTC server."""
        # Initialize GStreamer
        Gst.init(None)
        
        print("="*70)
        print("Starting DepthSplat WebRTC Server")
        print("="*70)
        print(f"WebRTC signaling port: {self.config.webrtc_port}")
        print(f"STUN server: {self.config.stun_server}")
        print(f"Input resolution: {self.config.input_width}x{self.config.input_height}")
        print(f"Target resolution: {self.config.target_width}x{self.config.target_height}")
        print(f"Frame skip: {self.config.frame_skip}")
        print(f"Max FPS: {self.config.max_fps if self.config.max_fps > 0 else 'unlimited'}")
        if self.config.camera_input_dir:
            print(f"Camera input save: {self.config.camera_input_dir}")
        print("="*70)
        
        # Initialize output sequencer
        def on_output_complete(total_emitted: int):
            self.frame_pairs_processed = total_emitted
            if self.config.max_frame_pairs > 0 and total_emitted >= self.config.max_frame_pairs:
                print(f"\n[Complete] Processed {total_emitted} frame pairs")
                self._request_stop()
        
        self.output_sequencer = OutputSequencer(
            self.config,
            on_complete=on_output_complete,
            save_files=self.config.save_files,
        )
        
        # Initialize GPU workers
        self.worker_pool = GPUWorkerPool(self.config)
        self.worker_pool.start()
        
        print("\n" + "="*70)
        print("Waiting for encoder initialization...")
        print("="*70)
        if not self.worker_pool.wait_all_ready(timeout=300):
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
        
        # Create WebRTC camera pipelines
        self.cameras = {
            0: WebRTCCameraPipeline(
                camera_id=0,
                config=self.config,
                frame_callback=self._on_frame,
            ),
            1: WebRTCCameraPipeline(
                camera_id=1,
                config=self.config,
                frame_callback=self._on_frame,
            ),
        }
        
        # Start camera pipelines
        for camera in self.cameras.values():
            camera.start()
        
        # Start signaling server
        self.signaling_server = WebRTCSignalingServer(self.config, self.cameras)
        self.signaling_server.start()
        
        # Start web client server (serves the HTML page over HTTP)
        # Note: Only WebSocket signaling uses SSL, not the HTML page server
        self.web_client_server = WebClientServer(
            port=self.config.web_client_port,
            verbose=self.config.verbose,
            signaling_port=self.config.webrtc_port,
        )
        self.web_client_server.start()
        
        # Set up duration timeout if configured
        if self.config.duration_seconds > 0:
            def on_timeout():
                print(f"\n[Timer] Duration of {self.config.duration_seconds}s reached")
                self._request_stop()
                return False
            GLib.timeout_add(int(self.config.duration_seconds * 1000), on_timeout)
        
        # Status reporting
        def report_status():
            if not self.running:
                return False
            elapsed = time.time() - self.start_time
            cam0_status = "✓" if self.cameras[0].connected else "✗"
            cam1_status = "✓" if self.cameras[1].connected else "✗"
            
            # Show actual in-flight count (pull model ensures this stays small)
            in_flight = self.worker_pool.get_in_flight_count()
            dropped = self.frame_pairs_queued - self.frame_pairs_processed - in_flight
            if dropped < 0:
                dropped = 0  # Can be negative briefly during transitions
            
            print(f"[Status] Elapsed: {elapsed:.1f}s | Cameras: [{cam0_status}|{cam1_status}] | InFlight: {in_flight} | Processed: {self.frame_pairs_processed} | Dropped: {dropped}")
            return True
        GLib.timeout_add(5000, report_status)
        
        # Print connection instructions
        is_ssl = bool(self.config.ssl_cert and self.config.ssl_key)
        ws_proto = "wss" if is_ssl else "ws"
        
        print("\n" + "="*70)
        print("WebRTC Server Ready" + (" (WSS Enabled 🔒)" if is_ssl else ""))
        print("="*70)
        print(f"\n📱 Open the client in your browser:")
        print(f"   http://YOUR_IP:{self.config.web_client_port}")
        print(f"\n🔌 Signaling WebSocket:")
        print(f"   {ws_proto}://YOUR_IP:{self.config.webrtc_port}")
        if not is_ssl:
            print(f"\n⚠️  For iOS/mobile: Use --generate-ssl for secure WebSocket")
        else:
            print(f"\n✅ WSS enabled for signaling - iOS/mobile camera access should work")
            print(f"   (Accept the self-signed certificate warning when connecting)")
        print("="*70)
        print("\nServer running. Press Ctrl+C to stop.\n")
        
        # Run main loop
        self.main_loop = GLib.MainLoop()
        try:
            self.main_loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def _request_stop(self):
        """Request the server to stop."""
        if self.main_loop and self.main_loop.is_running():
            GLib.idle_add(self.main_loop.quit)
    
    def stop(self):
        """Stop the server."""
        print("\nStopping WebRTC server...")
        
        self.running = False
        
        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()
        
        # Stop web client server
        if self.web_client_server:
            self.web_client_server.stop()
        
        # Stop signaling server
        if self.signaling_server:
            self.signaling_server.stop()
        
        # Stop cameras
        for camera in self.cameras.values():
            camera.stop()
        
        # Stop worker pool
        if self.worker_pool:
            self.worker_pool.stop()
        
        # Stop process thread
        if self.process_thread:
            self.latest_frame_event.set()  # Wake up the thread so it can exit
            self.process_thread.join(timeout=5.0)
        
        print("WebRTC server stopped")
    
    def _on_frame(self, camera_id: int, timestamp_ns: int, frame: np.ndarray):
        """Handle incoming frame from a WebRTC camera."""
        if not self.running:
            return
        
        if self.config.max_frame_pairs > 0 and self.frame_pairs_queued >= self.config.max_frame_pairs:
            return
        
        # Synchronize with other camera
        result = self.synchronizer.add_frame(camera_id, timestamp_ns, frame)
        
        if result is not None:
            sync_seq_id, frames_dict = result
            
            # Rate limiting
            if self.min_frame_interval > 0:
                current_time = time.time()
                elapsed = current_time - self.last_process_time
                if elapsed < self.min_frame_interval:
                    return
                self.last_process_time = current_time
            
            # Assign sequential processing ID
            with self.seq_counter_lock:
                proc_seq_id = self.processing_seq_counter
                self.processing_seq_counter += 1
            
            # Store as latest frame (overwrites any previous unprocessed frame)
            with self.latest_frame_lock:
                old_frame = self.latest_frame
                self.latest_frame = (proc_seq_id, frames_dict)
                self.frame_pairs_queued += 1
                
            
            # Signal that a new frame is available
            self.latest_frame_event.set()
    
    def _process_loop(self):
        """
        Process the latest frame pair using pull-based model.
        
        This waits for a worker to become available FIRST, then grabs the latest
        frame. This ensures we always process the freshest possible frame and
        keeps queue depth = num_workers (like RTMP/SRT low-latency modes).
        """
        while self.running:
            # PULL MODEL: Wait for a worker to become available first
            # This ensures queue depth never exceeds num_workers
            worker = self.worker_pool.wait_for_available_worker(timeout=1.0)
            if worker is None:
                continue  # Timeout, check if still running
            
            if not self.running:
                break
            
            # Now that we have an available worker, get the latest frame
            # Wait for a frame if none available yet
            while self.running:
                with self.latest_frame_lock:
                    if self.latest_frame is not None:
                        seq_id, frames_dict = self.latest_frame
                        self.latest_frame = None
                        self.latest_frame_event.clear()
                        break
                
                # No frame yet, wait for one
                if not self.latest_frame_event.wait(timeout=0.5):
                    continue
            else:
                # Server stopped
                break
            
            # Mark as submitted for correct output ordering, then submit to specific worker
            self.output_sequencer.mark_submitted(seq_id)
            self.worker_pool.submit_to_worker(
                worker,
                seq_id,
                frames_dict,
                self.output_sequencer.add_result,
                self.output_sequencer.cancel_submitted
            )


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
        
        # Latest frame buffer (always process newest frame for lowest latency)
        self.latest_frame: Optional[tuple] = None  # Holds (seq_id, frames_dict) or None
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_event = threading.Event()  # Signals when new frame available
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
        if self.config.camera_input_dir:
            print(f"Camera input save: {self.config.camera_input_dir}")
        print("="*70)
        
        # Initialize components
        def on_output_complete(total_emitted: int):
            self.frame_pairs_processed = total_emitted
            # Check if we should stop after processing max_frame_pairs
            if self.config.max_frame_pairs > 0 and total_emitted >= self.config.max_frame_pairs:
                print(f"\n[Complete] Processed {total_emitted} frame pairs")
                self._request_stop()
        
        self.output_sequencer = OutputSequencer(
            self.config,
            on_complete=on_output_complete,
            save_files=self.config.save_files,
        )
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
            # Get pipeline states
            cam_states = []
            for cam in self.cameras:
                state = cam.get_pipeline_state()
                frames = cam.frame_count
                cam_states.append(f"cam{cam.camera_id}:{state}({frames}f)")
            states_str = " | ".join(cam_states)
            
            # Show actual in-flight count (pull model ensures this stays small)
            in_flight = self.worker_pool.get_in_flight_count()
            dropped = self.frame_pairs_queued - self.frame_pairs_processed - in_flight
            if dropped < 0:
                dropped = 0  # Can be negative briefly during transitions
            
            print(f"[Status] Elapsed: {elapsed:.1f}s | InFlight: {in_flight} | Processed: {self.frame_pairs_processed} | Dropped: {dropped} | {states_str}")
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
            self.latest_frame_event.set()  # Wake up the thread so it can exit
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
            
            # Store as latest frame (overwrites any previous unprocessed frame)
            with self.latest_frame_lock:
                old_frame = self.latest_frame
                self.latest_frame = (proc_seq_id, frames_dict)
                self.frame_pairs_queued += 1
                
            
            # Signal that a new frame is available
            self.latest_frame_event.set()
            
            # Check if we've reached the max and should stop accepting new frames
            if self.config.max_frame_pairs > 0 and self.frame_pairs_queued >= self.config.max_frame_pairs:
                print(f"[Limit] Queued {self.config.max_frame_pairs} frame pairs, waiting for processing...")
                    
    def _process_loop(self):
        """
        Process the latest frame pair using pull-based model.
        
        This waits for a worker to become available FIRST, then grabs the latest
        frame. This ensures we always process the freshest possible frame and
        keeps queue depth = num_workers (like RTMP/SRT low-latency modes).
        """
        while self.running:
            # PULL MODEL: Wait for a worker to become available first
            # This ensures queue depth never exceeds num_workers
            worker = self.worker_pool.wait_for_available_worker(timeout=1.0)
            if worker is None:
                continue  # Timeout, check if still running
            
            if not self.running:
                break
            
            # Now that we have an available worker, get the latest frame
            # Wait for a frame if none available yet
            while self.running:
                with self.latest_frame_lock:
                    if self.latest_frame is not None:
                        seq_id, frames_dict = self.latest_frame
                        self.latest_frame = None
                        self.latest_frame_event.clear()
                        break
                
                # No frame yet, wait for one
                if not self.latest_frame_event.wait(timeout=0.5):
                    continue
            else:
                # Server stopped
                break
            
            # Save input frames if debugging
            if self.config.save_input_frames:
                self._save_debug_frames(seq_id, frames_dict)
            
            # Mark as submitted for correct output ordering, then submit to specific worker
            self.output_sequencer.mark_submitted(seq_id)
            self.worker_pool.submit_to_worker(
                worker,
                seq_id,
                frames_dict,
                self.output_sequencer.add_result,
                self.output_sequencer.cancel_submitted
            )
                
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

def load_test_images(image_dir: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load test images from a directory.
    
    Args:
        image_dir: Directory containing images (jpg/png files)
        
    Returns:
        List of (image, metadata_dict) tuples where image is HWC uint8 numpy array
    """
    from PIL import Image
    import json
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(sorted(image_dir.glob(ext)))
    
    if not image_files:
        raise ValueError(f"No image files found in: {image_dir}")
    
    results = []
    for img_path in image_files:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.uint8)
        results.append((img_array, str(img_path)))
    
    return results


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


def create_file_test_config(image_dir: str) -> ServerConfig:
    """
    Create a test configuration for file-based testing.
    
    Args:
        image_dir: Directory containing test images
    """
    return ServerConfig(
        # These won't be used in file mode
        camera_0_source="",
        camera_1_source="",
        input_width=960,
        input_height=512,
        input_framerate=1,
        max_frame_pairs=1,  # Process one pair
        verbose=True,
        test_images_dir=image_dir,
    )


def run_file_inference(
    image_dir: str,
    output_dir: str = "stream-output",
    verbose: bool = True,
    num_iterations: int = 1,
    warmup_iterations: int = 1,
    output_format: str = "spz",
    output_callback: Optional[Callable[[int, bytes], None]] = None,
    save_files: bool = True,
) -> bytes:
    """
    Run inference on images from a directory with optional multiple iterations for benchmarking.
    
    This is a simplified function for testing with file-based images
    without starting the full GStreamer server.
    
    Args:
        image_dir: Directory containing at least 2 images
        output_dir: Directory to save output file
        verbose: Print progress
        num_iterations: Number of inference iterations to run for benchmarking (default: 1)
        warmup_iterations: Number of warmup iterations before timing (default: 1)
        output_format: Output format - "spz" (compressed, default) or "ply"
        output_callback: Optional callback for streaming output (seq_id, bytes)
        save_files: Whether to save output files to disk (default: True)
        
    Returns:
        Output file contents as bytes (from the last iteration)
    """
    from PIL import Image
    
    total_start = time.time()
    
    # Load images
    load_start = time.time()
    images_data = load_test_images(image_dir)
    load_elapsed = time.time() - load_start
    
    if len(images_data) < 2:
        raise ValueError(f"Need at least 2 images for inference, found {len(images_data)}")
    
    # Take first 2 images
    frame_0, path_0 = images_data[0]
    frame_1, path_1 = images_data[1]
    
    if verbose:
        print(f"Using images:")
        print(f"  Camera 0: {path_0} ({frame_0.shape})")
        print(f"  Camera 1: {path_1} ({frame_1.shape})")
        print(f"  Image loading time: {load_elapsed:.3f}s")
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"\nInitializing DepthAnything3...")
    
    da3_init_start = time.time()
    depth_anything = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    depth_anything = depth_anything.to(device=device)
    depth_anything.eval()
    da3_init_elapsed = time.time() - da3_init_start
    
    if verbose:
        print(f"  DepthAnything3 init time: {da3_init_elapsed:.3f}s")
        print(f"\nInitializing DepthSplat encoder...")
    
    encoder_init_start = time.time()
    encoder = DepthSplatInference()
    encoder_init_elapsed = time.time() - encoder_init_start
    
    if verbose:
        print(f"  DepthSplat encoder init time: {encoder_init_elapsed:.3f}s")
    
    # Get original dimensions
    orig_height, orig_width = frame_0.shape[:2]
    
    # Pre-convert frames to tensors (done once, reused for all iterations)
    frame_0_tensor = torch.from_numpy(frame_0).permute(2, 0, 1).float() / 255.0
    frame_1_tensor = torch.from_numpy(frame_1).permute(2, 0, 1).float() / 255.0
    images = torch.stack([frame_0_tensor, frame_1_tensor], dim=0)
    
    # Inference configuration
    inference_config = InferenceConfig(
        target_height=512,
        target_width=960,
        near_disparity=1.0,
        far_disparity=0.1,
        verbose=False,  # Suppress per-iteration verbose output
        output_format=output_format,
    )
    
    # Storage for timing statistics
    da3_times = []
    encoder_times = []
    total_inference_times = []
    
    total_iterations = warmup_iterations + num_iterations
    
    if verbose:
        print(f"\n" + "="*70)
        print(f"BENCHMARK: {warmup_iterations} warmup + {num_iterations} timed iterations")
        print(f"="*70)
    
    output_bytes = None
    
    for iteration in range(total_iterations):
        is_warmup = iteration < warmup_iterations
        iter_label = f"Warmup {iteration + 1}/{warmup_iterations}" if is_warmup else f"Iteration {iteration - warmup_iterations + 1}/{num_iterations}"
        
        if verbose:
            print(f"\n[{iter_label}]")
        
        iter_start = time.time()
        
        # Run DepthAnything3
        da3_start = time.time()
        prediction = depth_anything.inference(
            image=[frame_0, frame_1],
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        
        # Ensure CUDA synchronization for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        da3_elapsed = time.time() - da3_start
        
        # Get processed size (only need to do this once)
        processed_height, processed_width = prediction.processed_images.shape[1:3]
        
        # Convert W2C to C2W
        w2c = prediction.extrinsics
        c2w = w2c_to_c2w(w2c)
        
        # Scale intrinsics
        da3_intrinsics = prediction.intrinsics
        scaled_intrinsics = scale_intrinsics(
            da3_intrinsics,
            from_width=processed_width,
            from_height=processed_height,
            to_width=orig_width,
            to_height=orig_height,
        )
        
        # Run encoder
        encoder_start = time.time()
        output_bytes = encoder.run_from_data(
            images=images,
            intrinsics_list=[scaled_intrinsics[0], scaled_intrinsics[1]],
            extrinsics_list=[c2w[0], c2w[1]],
            original_sizes=[(orig_width, orig_height), (orig_width, orig_height)],
            image_filenames=[Path(path_0).name, Path(path_1).name],
            config=inference_config,
        )
        
        # Ensure CUDA synchronization for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encoder_elapsed = time.time() - encoder_start
        
        iter_elapsed = time.time() - iter_start
        
        if verbose:
            status = "(warmup - not counted)" if is_warmup else ""
            print(f"  DepthAnything3: {da3_elapsed:.3f}s | Encoder: {encoder_elapsed:.3f}s | Total: {iter_elapsed:.3f}s {status}")
        
        # Only record times for non-warmup iterations
        if not is_warmup:
            da3_times.append(da3_elapsed)
            encoder_times.append(encoder_elapsed)
            total_inference_times.append(iter_elapsed)
            
            # Call output callback if provided (for WebSocket streaming, etc.)
            if output_callback:
                seq_id = iteration - warmup_iterations
                try:
                    output_callback(seq_id, output_bytes)
                except Exception as e:
                    print(f"Output callback error for seq {seq_id}: {e}")
    
    # Save output (from last iteration) if file saving is enabled
    output_file = None
    if save_files:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ext = output_format.lower()
        output_file = output_path / f"gaussians.{ext}"
        
        with open(output_file, 'wb') as f:
            f.write(output_bytes)
    
    total_elapsed = time.time() - total_start
    
    # Compute statistics
    def compute_stats(times: list) -> dict:
        if not times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        arr = np.array(times)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    da3_stats = compute_stats(da3_times)
    encoder_stats = compute_stats(encoder_times)
    total_stats = compute_stats(total_inference_times)
    
    if verbose:
        print(f"\n" + "="*70)
        print(f"TIMING SUMMARY")
        print(f"="*70)
        print(f"Setup (one-time):")
        print(f"  Image loading:          {load_elapsed:.3f}s")
        print(f"  DepthAnything3 init:    {da3_init_elapsed:.3f}s")
        print(f"  DepthSplat init:        {encoder_init_elapsed:.3f}s")
        print(f"  ---------------------------------")
        print(f"  Total setup:            {load_elapsed + da3_init_elapsed + encoder_init_elapsed:.3f}s")
        
        if num_iterations > 1:
            print(f"\nInference Statistics ({num_iterations} iterations, excluding {warmup_iterations} warmup):")
            print(f"  {'Component':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print(f"  {'-'*60}")
            print(f"  {'DepthAnything3':<20} {da3_stats['mean']:>9.3f}s {da3_stats['std']:>9.3f}s {da3_stats['min']:>9.3f}s {da3_stats['max']:>9.3f}s")
            print(f"  {'DepthSplat Encoder':<20} {encoder_stats['mean']:>9.3f}s {encoder_stats['std']:>9.3f}s {encoder_stats['min']:>9.3f}s {encoder_stats['max']:>9.3f}s")
            print(f"  {'-'*60}")
            print(f"  {'Total Inference':<20} {total_stats['mean']:>9.3f}s {total_stats['std']:>9.3f}s {total_stats['min']:>9.3f}s {total_stats['max']:>9.3f}s")
            print(f"\n  Throughput: {1.0/total_stats['mean']:.2f} FPS (based on mean)")
        else:
            print(f"\nInference (single run):")
            print(f"  DepthAnything3:         {da3_stats['mean']:.3f}s")
            print(f"  DepthSplat Encoder:     {encoder_stats['mean']:.3f}s")
            print(f"  ---------------------------------")
            print(f"  Total inference:        {total_stats['mean']:.3f}s")
            print(f"  Throughput:             {1.0/total_stats['mean']:.2f} FPS")
        
        print(f"\n  Total wall time:        {total_elapsed:.3f}s")
        print(f"="*70)
        if output_file:
            print(f"\n✓ Saved {output_format.upper()} to {output_file}")
        if output_callback:
            print(f"\n✓ Streamed {num_iterations} frame(s) via callback")
        print(f"  File size: {len(output_bytes) / (1024*1024):.2f} MB")
    
    return output_bytes


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


def create_rtmp_config(url_0: str, url_1: str) -> ServerConfig:
    """
    Create a configuration for RTMP streams.
    
    Uses decodebin for automatic codec detection and dynamic pad handling,
    which is required for FLV container demuxing.
    
    Uses frame-count-based sync since RTMP streams from different devices
    have unsynchronized timestamps.
    
    Args:
        url_0: RTMP URL for camera 0 (e.g., rtmp://server/live/stream1)
        url_1: RTMP URL for camera 1 (e.g., rtmp://server/live/stream2)
    """
    return ServerConfig(
        camera_0_source=f"rtmpsrc location={url_0} ! decodebin ! videoconvert",
        camera_1_source=f"rtmpsrc location={url_1} ! decodebin ! videoconvert",
        input_width=1920,
        input_height=1080,
        input_framerate=30,
        frame_skip=2,
        max_fps=15.0,
        use_frame_count_sync=True,  # Use frame count sync since RTMP devices have different clocks
        verbose=True,
    )


def create_srt_config(url_0: str, url_1: str) -> ServerConfig:
    """
    Create a configuration for SRT (Secure Reliable Transport) streams.
    
    SRT provides low-latency, reliable video transport with built-in encryption.
    Uses srtsrc element with decodebin for automatic codec detection.
    
    Uses frame-count-based sync since SRT streams from different devices
    have unsynchronized timestamps.
    
    Note: Each camera requires a separate port. GStreamer's srtsrc elements
    cannot share a port even with different streamids (they conflict at the
    socket level).
    
    URI formats:
        - Listener mode (server waits for connection): srt://:8080?mode=listener
        - Caller mode (connect to sender): srt://host:8080?mode=caller
        - Default mode (caller): srt://host:8080
    
    Args:
        url_0: SRT URL for camera 0 (e.g., srt://:8080?mode=listener)
        url_1: SRT URL for camera 1 (e.g., srt://:8081?mode=listener)
    """
    return ServerConfig(
        camera_0_source=f'srtsrc uri="{url_0}" ! decodebin ! videoconvert',
        camera_1_source=f'srtsrc uri="{url_1}" ! decodebin ! videoconvert',
        input_width=1920,
        input_height=1080,
        input_framerate=30,
        frame_skip=2,
        max_fps=15.0,
        use_frame_count_sync=True,  # Use frame count sync since SRT devices have different clocks
        verbose=True,
    )


def create_webrtc_config(
    port: int = 8080,
    stun_server: str = "stun://stun.l.google.com:19302",
) -> ServerConfig:
    """
    Create a configuration for WebRTC camera input.
    
    Args:
        port: Port for WebRTC signaling server
        stun_server: STUN server URL for ICE
    """
    return ServerConfig(
        input_width=1280,
        input_height=720,
        input_framerate=30,
        webrtc_port=port,
        stun_server=stun_server,
        frame_skip=1,
        max_fps=10.0,
        verbose=True,
    )


def create_tcp_h264_config(
    host_0: str = "0.0.0.0",
    port_0: int = 5000,
    host_1: str = "0.0.0.0",
    port_1: int = 5001,
    listen: bool = True,
    stream_format: str = "auto",
) -> ServerConfig:
    """
    Create a configuration for H.264 video over TCP streams.
    
    This mode receives H.264 video streams over TCP connections.
    Each camera requires a separate TCP connection on a different port.
    
    Args:
        host_0: Host address for camera 0 (IP to bind/connect)
        port_0: TCP port for camera 0
        host_1: Host address for camera 1 (IP to bind/connect)
        port_1: TCP port for camera 1
        listen: If True, use tcpserversrc (listen for connections).
                If False, use tcpclientsrc (connect to sender).
        stream_format: Container/stream format:
            - "auto": Auto-detect using decodebin (default, most flexible)
            - "h264": Raw H.264 Annex B bitstream (NAL units with start codes)
            - "ts" or "mpegts": MPEG-TS container (common for streaming)
            - "flv": FLV container
                
    Note:
        - In listener mode (listen=True), the server waits for senders to connect.
          Good for mobile devices that initiate connections.
        - In client mode (listen=False), the server connects to senders.
          Good when senders have known addresses.
          
    Sender Examples (sending H.264 to this server):
        # Raw H.264 (--tcp-format h264):
        ffmpeg -i input.mp4 -c:v libx264 -tune zerolatency -f h264 tcp://server_ip:5000
        
        # MPEG-TS format (--tcp-format ts) - recommended for streaming:
        ffmpeg -i input.mp4 -c:v libx264 -tune zerolatency -f mpegts tcp://server_ip:5000
        
        # From GStreamer (raw H.264):
        gst-launch-1.0 videotestsrc ! x264enc tune=zerolatency ! tcpclientsink host=server_ip port=5000
        
        # From GStreamer (MPEG-TS):
        gst-launch-1.0 videotestsrc ! x264enc tune=zerolatency ! mpegtsmux ! tcpclientsink host=server_ip port=5000
    """
    # Build decoder pipeline based on format
    if stream_format == "h264" or stream_format == "raw":
        # Raw H.264 Annex B bitstream
        decoder = "h264parse ! avdec_h264 ! videoconvert"
    elif stream_format == "ts" or stream_format == "mpegts":
        # MPEG-TS container
        decoder = "tsdemux ! h264parse ! avdec_h264 ! videoconvert"
    elif stream_format == "flv":
        # FLV container
        decoder = "flvdemux ! h264parse ! avdec_h264 ! videoconvert"
    else:  # "auto" or anything else
        # Auto-detect using decodebin
        decoder = "decodebin ! videoconvert"
    
    if listen:
        # Server mode - listen for incoming connections
        # do-timestamp=true: Generate timestamps for incoming data
        source_0 = f"tcpserversrc host={host_0} port={port_0} do-timestamp=true ! queue ! {decoder}"
        source_1 = f"tcpserversrc host={host_1} port={port_1} do-timestamp=true ! queue ! {decoder}"
    else:
        # Client mode - connect to senders
        source_0 = f"tcpclientsrc host={host_0} port={port_0} do-timestamp=true ! queue ! {decoder}"
        source_1 = f"tcpclientsrc host={host_1} port={port_1} do-timestamp=true ! queue ! {decoder}"
    
    return ServerConfig(
        camera_0_source=source_0,
        camera_1_source=source_1,
        input_width=1920,
        input_height=1080,
        input_framerate=30,
        frame_skip=2,
        max_fps=15.0,
        use_frame_count_sync=False,  # Use timestamp sync - TCP streams have valid timestamps after first frame
        verbose=True,
    )


def generate_self_signed_cert(
    ssl_dir: str = "/workspace/ssl",
    cert_name: str = "cert.pem",
    key_name: str = "key.pem",
    days: int = 365,
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Generate a self-signed SSL certificate for HTTPS/WSS.
    
    Args:
        ssl_dir: Directory to store certificates
        cert_name: Certificate filename
        key_name: Private key filename
        days: Certificate validity in days
        verbose: Print progress messages
        
    Returns:
        Tuple of (cert_path, key_path)
    """
    ssl_path = Path(ssl_dir)
    ssl_path.mkdir(parents=True, exist_ok=True)
    
    cert_path = ssl_path / cert_name
    key_path = ssl_path / key_name
    
    # Check if certificates already exist
    if cert_path.exists() and key_path.exists():
        if verbose:
            print(f"[SSL] Using existing certificates in {ssl_dir}")
        return str(cert_path), str(key_path)
    
    if verbose:
        print(f"[SSL] Generating self-signed certificate...")
    
    # Generate using openssl
    cmd = [
        "openssl", "req", "-x509",
        "-newkey", "rsa:4096",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-days", str(days),
        "-nodes",  # No password
        "-subj", "/CN=localhost/O=DepthSplat/C=US",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if verbose:
            print(f"[SSL] Certificate generated:")
            print(f"      Cert: {cert_path}")
            print(f"      Key:  {key_path}")
        return str(cert_path), str(key_path)
    except subprocess.CalledProcessError as e:
        print(f"[SSL] Failed to generate certificate: {e.stderr}")
        raise RuntimeError(f"Failed to generate SSL certificate: {e}")
    except FileNotFoundError:
        print("[SSL] Error: openssl not found. Install with: apt install openssl")
        raise RuntimeError("openssl command not found")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DepthSplat GStreamer Server with DepthAnything3 camera estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File-based test mode (recommended for testing):
  python server.py --mode file --image-dir /workspace/input_images
  
  # File-based benchmarking with 10 iterations and 2 warmup runs:
  python server.py --mode file --image-dir /workspace/input_images --iterations 10 --warmup 2
  
  # GStreamer test mode - process 5 frame pairs then exit:
  python server.py --mode test --num-frames 5
  
  # GStreamer test mode - run for 30 seconds:
  python server.py --mode test --duration 30
  
  # V4L2 cameras with throttling:
  python server.py --mode v4l2 --frame-skip 2 --max-fps 5
  
  # RTSP streams:
  python server.py --mode rtsp --rtsp0 rtsp://cam1/stream --rtsp1 rtsp://cam2/stream
  
  # RTMP streams:
  python server.py --mode rtmp --rtmp0 rtmp://server/live/cam1 --rtmp1 rtmp://server/live/cam2
  
  # SRT streams (listener mode - iOS devices send to server):
  python server.py --mode srt --srt-listen
  
  # SRT streams (listener mode - custom ports):
  python server.py --mode srt --srt-listen 9000,9001
  
  # SRT streams (caller mode - connect to senders):
  python server.py --mode srt --srt0 srt://192.168.1.100:8080 --srt1 srt://192.168.1.100:8081
  
  # TCP H.264 streams (listener mode - wait for senders to connect):
  python server.py --mode tcp --tcp-listen
  
  # TCP H.264 with MPEG-TS format (recommended for streaming):
  python server.py --mode tcp --tcp-listen --tcp-format ts
  
  # TCP H.264 with raw H.264 format:
  python server.py --mode tcp --tcp-listen --tcp-format h264
  
  # TCP H.264 streams (listener mode - custom ports):
  python server.py --mode tcp --tcp-listen 6000,6001
  
  # TCP H.264 streams (client mode - connect to senders):
  python server.py --mode tcp --tcp0 192.168.1.100:5000 --tcp1 192.168.1.100:5001
  
  # WebRTC mode - receive streams from browsers/mobile apps:
  python server.py --mode webrtc --webrtc-port 8080
  
  # WebRTC with SSL (required for iOS camera access):
  python server.py --mode webrtc --generate-ssl
  
  # WebRTC with custom SSL certificates:
  python server.py --mode webrtc --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem
  
  # WebRTC with custom STUN server and output streaming:
  python server.py --mode webrtc --webrtc-port 8080 --stun-server stun://mystun.server:3478 --websocket
"""
    )
    parser.add_argument(
        "--mode",
        choices=["test", "file", "v4l2", "rtsp", "rtmp", "srt", "tcp", "webrtc", "custom"],
        default="file",
        help="Camera mode (default: file)"
    )
    parser.add_argument(
        "--image-dir",
        default="/workspace/input_images",
        help="Directory containing test images for file mode (default: /workspace/input_images)"
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
        "--rtmp0",
        help="RTMP URL for camera 0 (e.g., rtmp://server/live/stream1)"
    )
    parser.add_argument(
        "--rtmp1",
        help="RTMP URL for camera 1 (e.g., rtmp://server/live/stream2)"
    )
    parser.add_argument(
        "--srt0",
        help="SRT URL for camera 0 (e.g., srt://host:8080 or srt://:8080?mode=listener)"
    )
    parser.add_argument(
        "--srt1",
        help="SRT URL for camera 1 (e.g., srt://host:8081 or srt://:8081?mode=listener)"
    )
    parser.add_argument(
        "--srt-listen",
        nargs="?",
        const="8080,8081",
        metavar="PORT0,PORT1",
        help="SRT listener mode: wait for connections on specified ports (default: 8080,8081)"
    )
    
    # TCP H.264 mode arguments
    parser.add_argument(
        "--tcp0",
        help="TCP address for camera 0 in client mode (e.g., 192.168.1.100:5000)"
    )
    parser.add_argument(
        "--tcp1",
        help="TCP address for camera 1 in client mode (e.g., 192.168.1.100:5001)"
    )
    parser.add_argument(
        "--tcp-listen",
        nargs="?",
        const="5000,5001",
        metavar="PORT0,PORT1",
        help="TCP listener mode: wait for H.264 streams on specified ports (default: 5000,5001)"
    )
    parser.add_argument(
        "--tcp-format",
        choices=["auto", "h264", "ts", "mpegts", "flv"],
        default="auto",
        help="TCP stream format: auto (default), h264 (raw Annex B), ts/mpegts (MPEG-TS container), flv"
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations for benchmarking in file mode (default: 1)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations before timing in file mode (default: 1)"
    )
    parser.add_argument(
        "--format",
        choices=["spz", "ply"],
        default="spz",
        help="Output format: 'spz' (compressed, default) or 'ply' (standard)"
    )
    
    # WebSocket streaming arguments
    parser.add_argument(
        "--websocket",
        action="store_true",
        help="Enable WebSocket streaming to remote clients"
    )
    parser.add_argument(
        "--ws-host",
        default="0.0.0.0",
        help="WebSocket server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--ws-only",
        action="store_true",
        help="Only stream via WebSocket, don't save files to disk"
    )
    
    # Camera input save argument
    parser.add_argument(
        "--save-camera-input",
        nargs="?",
        const="camera-input",
        default=None,
        metavar="DIR",
        help="Save camera input frames to disk before inference. Optional directory name (default: camera-input)"
    )
    
    # WebRTC mode arguments
    parser.add_argument(
        "--webrtc-port",
        type=int,
        default=8080,
        help="Port for WebRTC signaling server (default: 8080)"
    )
    parser.add_argument(
        "--stun-server",
        default="stun://stun.l.google.com:19302",
        help="STUN server URL for WebRTC ICE (default: stun://stun.l.google.com:19302)"
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8888,
        help="Port for serving the WebRTC client HTML page (default: 8888)"
    )
    
    # SSL/TLS arguments
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="Path to SSL certificate file (enables HTTPS/WSS)"
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=None,
        help="Path to SSL private key file"
    )
    parser.add_argument(
        "--ssl-dir",
        type=str,
        default="/workspace/ssl",
        help="Directory to store/find SSL certificates (default: /workspace/ssl)"
    )
    parser.add_argument(
        "--generate-ssl",
        action="store_true",
        help="Generate self-signed SSL certificates if they don't exist"
    )
    
    args = parser.parse_args()
    
    # File-based test mode - runs without GStreamer
    if args.mode == "file":
        print("="*70)
        print("File-based Test Mode")
        print("="*70)
        print(f"Image directory: {args.image_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output format: {args.format.upper()}")
        print(f"Iterations: {args.iterations} (+ {args.warmup} warmup)")
        
        # Initialize WebSocket streamer if enabled
        websocket_streamer = None
        output_callback = None
        save_files = True
        
        if args.websocket:
            try:
                from websocket_streamer import WebSocketStreamer
                
                websocket_streamer = WebSocketStreamer(
                    host=args.ws_host,
                    port=args.ws_port,
                    output_format=args.format,
                    verbose=not args.quiet,
                )
                websocket_streamer.start()
                
                output_callback = websocket_streamer.broadcast_frame
                
                print(f"\nWebSocket streaming enabled:")
                print(f"  URL: ws://{args.ws_host}:{args.ws_port}")
                if args.ws_only:
                    print(f"  Mode: WebSocket only (no files saved)")
                    save_files = False
                else:
                    print(f"  Mode: WebSocket + file output")
                    
            except ImportError as e:
                print(f"Warning: Could not enable WebSocket streaming: {e}")
                print("Install websockets with: pip install websockets")
        
        print("="*70 + "\n")
        
        try:
            run_file_inference(
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                verbose=not args.quiet,
                num_iterations=args.iterations,
                warmup_iterations=args.warmup,
                output_format=args.format,
                output_callback=output_callback,
                save_files=save_files,
            )
        finally:
            if websocket_streamer:
                websocket_streamer.stop()
        return
    
    # WebRTC mode - receive camera streams via WebRTC
    if args.mode == "webrtc":
        # Handle SSL certificates
        ssl_cert = args.ssl_cert
        ssl_key = args.ssl_key
        
        # Generate self-signed certificates if requested
        if args.generate_ssl or (ssl_cert is None and ssl_key is None):
            # Check if certs exist in ssl_dir
            ssl_dir = Path(args.ssl_dir)
            default_cert = ssl_dir / "cert.pem"
            default_key = ssl_dir / "key.pem"
            
            if args.generate_ssl or (default_cert.exists() and default_key.exists()):
                try:
                    ssl_cert, ssl_key = generate_self_signed_cert(
                        ssl_dir=args.ssl_dir,
                        verbose=not args.quiet,
                    )
                except Exception as e:
                    if args.generate_ssl:
                        print(f"Error generating SSL certificates: {e}")
                        return
                    # If not explicitly requested, just continue without SSL
                    ssl_cert = None
                    ssl_key = None
        
        config = ServerConfig(
            input_width=1280,
            input_height=720,
            webrtc_port=args.webrtc_port,
            web_client_port=args.web_port,
            stun_server=args.stun_server,
            ssl_cert=ssl_cert,
            ssl_key=ssl_key,
            output_dir=args.output_dir,
            output_format=args.format,
            verbose=not args.quiet,
        )
        
        # Apply overrides
        if args.frame_skip is not None:
            config.frame_skip = args.frame_skip
        if args.max_fps is not None:
            config.max_fps = args.max_fps
        if args.duration > 0:
            config.duration_seconds = args.duration
        if args.num_frames > 0:
            config.max_frame_pairs = args.num_frames
        if args.save_camera_input is not None:
            config.camera_input_dir = args.save_camera_input
        
        # Initialize WebSocket output streamer if enabled
        websocket_streamer = None
        if args.websocket:
            try:
                from websocket_streamer import WebSocketStreamer
                
                websocket_streamer = WebSocketStreamer(
                    host=args.ws_host,
                    port=args.ws_port,
                    output_format=args.format,
                    verbose=not args.quiet,
                )
                websocket_streamer.start()
                config.output_callback = websocket_streamer.broadcast_frame
                
                if args.ws_only:
                    config.save_files = False
                    
            except ImportError as e:
                print(f"Warning: Could not enable WebSocket output streaming: {e}")
        
        # Start WebRTC server
        server = WebRTCDepthSplatServer(config)
        try:
            server.start()
        finally:
            if websocket_streamer:
                websocket_streamer.stop()
        return
    
    # GStreamer-based modes
    # Create configuration based on mode
    if args.mode == "test":
        config = create_test_config(num_frames=args.num_frames, duration=args.duration)
    elif args.mode == "v4l2":
        config = create_v4l2_config(args.device0, args.device1)
    elif args.mode == "rtsp":
        if not args.rtsp0 or not args.rtsp1:
            parser.error("RTSP mode requires --rtsp0 and --rtsp1 URLs")
        config = create_rtsp_config(args.rtsp0, args.rtsp1)
    elif args.mode == "rtmp":
        if not args.rtmp0 or not args.rtmp1:
            parser.error("RTMP mode requires --rtmp0 and --rtmp1 URLs")
        config = create_rtmp_config(args.rtmp0, args.rtmp1)
    elif args.mode == "srt":
        # Handle --srt-listen shorthand for listener mode (separate ports required)
        if args.srt_listen:
            ports = args.srt_listen.split(",")
            port0 = ports[0].strip() if len(ports) > 0 else "8080"
            port1 = ports[1].strip() if len(ports) > 1 else "8081"
            args.srt0 = f"srt://:{port0}?mode=listener"
            args.srt1 = f"srt://:{port1}?mode=listener"
        
        if not args.srt0 or not args.srt1:
            parser.error("SRT mode requires --srt0 and --srt1 URLs, or use --srt-listen for listener mode")
        config = create_srt_config(args.srt0, args.srt1)
        
        # Print connection info for SRT listener mode
        if "mode=listener" in args.srt0 or "mode=listener" in args.srt1:
            import socket
            try:
                # Get actual IP address (not just hostname)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "YOUR_SERVER_IP"
            
            # Extract ports from URLs
            import re
            port0_match = re.search(r':(\d+)', args.srt0)
            port1_match = re.search(r':(\d+)', args.srt1)
            port0 = port0_match.group(1) if port0_match else "8080"
            port1 = port1_match.group(1) if port1_match else "8081"
            
            print("\n" + "="*70)
            print("SRT LISTENER MODE - Connection URLs for iOS devices")
            print("="*70)
            print(f"\n📱 Camera 0 (iOS device 1) should connect to:")
            print(f"   srt://{local_ip}:{port0}")
            print(f"\n📱 Camera 1 (iOS device 2) should connect to:")
            print(f"   srt://{local_ip}:{port1}")
            print("\n" + "="*70 + "\n")
    elif args.mode == "tcp":
        # Handle --tcp-listen shorthand for listener mode
        listen_mode = False
        if args.tcp_listen:
            listen_mode = True
            ports = args.tcp_listen.split(",")
            port0 = int(ports[0].strip()) if len(ports) > 0 else 5000
            port1 = int(ports[1].strip()) if len(ports) > 1 else 5001
            host0 = "0.0.0.0"
            host1 = "0.0.0.0"
        elif args.tcp0 and args.tcp1:
            # Client mode - parse host:port
            def parse_tcp_addr(addr):
                if ":" in addr:
                    h, p = addr.rsplit(":", 1)
                    return h, int(p)
                else:
                    return addr, 5000
            host0, port0 = parse_tcp_addr(args.tcp0)
            host1, port1 = parse_tcp_addr(args.tcp1)
        else:
            parser.error("TCP mode requires --tcp0 and --tcp1 addresses, or use --tcp-listen for listener mode")
        
        config = create_tcp_h264_config(
            host_0=host0,
            port_0=port0,
            host_1=host1,
            port_1=port1,
            listen=listen_mode,
            stream_format=args.tcp_format,
        )
        
        # Print connection info for TCP listener mode
        if listen_mode:
            import socket
            try:
                # Get actual IP address (not just hostname)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "YOUR_SERVER_IP"
            
            # Determine FFmpeg output format based on stream_format
            fmt = args.tcp_format
            if fmt == "ts" or fmt == "mpegts":
                ffmpeg_fmt = "mpegts"
                fmt_desc = "MPEG-TS"
            elif fmt == "flv":
                ffmpeg_fmt = "flv"
                fmt_desc = "FLV"
            elif fmt == "h264":
                ffmpeg_fmt = "h264"
                fmt_desc = "Raw H.264"
            else:
                ffmpeg_fmt = "mpegts"  # Recommend MPEG-TS for auto mode
                fmt_desc = "Auto-detect (recommend MPEG-TS)"
            
            print("\n" + "="*70)
            print(f"TCP H.264 LISTENER MODE - Format: {fmt_desc}")
            print("="*70)
            print(f"\n📹 Camera 0 - connect to: tcp://{local_ip}:{port0}")
            print(f"📹 Camera 1 - connect to: tcp://{local_ip}:{port1}")
            print(f"\n--- Sender Examples (FFmpeg) ---")
            print(f"\n# Camera 0 - from webcam (macOS):")
            print(f"ffmpeg -f avfoundation -framerate 30 -i \"0\" \\")
            print(f"  -c:v libx264 -preset ultrafast -tune zerolatency \\")
            print(f"  -f {ffmpeg_fmt} tcp://{local_ip}:{port0}")
            print(f"\n# Camera 0 - from webcam (Linux):")
            print(f"ffmpeg -f v4l2 -framerate 30 -i /dev/video0 \\")
            print(f"  -c:v libx264 -preset ultrafast -tune zerolatency \\")
            print(f"  -f {ffmpeg_fmt} tcp://{local_ip}:{port0}")
            print(f"\n# Camera 1 - from file (for testing):")
            print(f"ffmpeg -re -stream_loop -1 -i video.mp4 \\")
            print(f"  -c:v libx264 -preset ultrafast -tune zerolatency \\")
            print(f"  -f {ffmpeg_fmt} tcp://{local_ip}:{port1}")
            print("\n" + "="*70 + "\n")
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
    config.output_format = args.format
    config.verbose = not args.quiet
    if args.save_camera_input is not None:
        config.camera_input_dir = args.save_camera_input
    
    # Initialize WebSocket streamer if enabled
    websocket_streamer = None
    if args.websocket:
        try:
            from websocket_streamer import WebSocketStreamer
            
            websocket_streamer = WebSocketStreamer(
                host=args.ws_host,
                port=args.ws_port,
                output_format=args.format,
                verbose=not args.quiet,
            )
            websocket_streamer.start()
            
            print(f"\n{'='*70}")
            print(f"WebSocket streaming enabled")
            print(f"  URL: ws://{args.ws_host}:{args.ws_port}")
            print(f"  Format: {args.format.upper()}")
            if args.ws_only:
                print(f"  Mode: WebSocket only (no files saved)")
            else:
                print(f"  Mode: WebSocket + file output")
            print(f"{'='*70}\n")
            
            # Set the WebSocket broadcast as the output callback
            config.output_callback = websocket_streamer.broadcast_frame
            
            # Disable file saving if --ws-only is specified
            if args.ws_only:
                config.save_files = False
                
        except ImportError as e:
            print(f"Warning: Could not enable WebSocket streaming: {e}")
            print("Install websockets with: pip install websockets")
    
    # Start server
    server = DepthSplatServer(config)
    try:
        server.start()
    finally:
        # Clean up WebSocket streamer
        if websocket_streamer:
            websocket_streamer.stop()


if __name__ == "__main__":
    main()

