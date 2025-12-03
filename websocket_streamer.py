"""
WebSocket Streaming Module for DepthSplat Gaussian Splat Output.

This module provides a WebSocket server that streams Gaussian splat data (SPZ/PLY)
to connected clients in real-time.

Protocol:
    - Server listens on configurable host:port
    - Clients connect via WebSocket
    - Server sends binary messages with format:
        [4 bytes: seq_id (big-endian uint32)]
        [4 bytes: format_type (0=SPZ, 1=PLY)]
        [4 bytes: data_length (big-endian uint32)]
        [N bytes: gaussian splat data]
    - Clients can send JSON control messages:
        {"type": "ping"} -> server responds with {"type": "pong"}
        {"type": "status"} -> server responds with current stats
        {"type": "pause"} -> pause streaming to this client
        {"type": "resume"} -> resume streaming to this client

Usage:
    # Start standalone WebSocket server (for testing)
    python websocket_streamer.py --host 0.0.0.0 --port 8765
    
    # Or integrate with server.py by setting output_callback
    from websocket_streamer import WebSocketStreamer
    
    streamer = WebSocketStreamer(host="0.0.0.0", port=8765)
    streamer.start()  # Non-blocking, runs in background
    
    config = ServerConfig(
        output_callback=streamer.broadcast_frame,
        ...
    )

Client Implementation Notes:
    - Swift (Mac/Vision Pro): Use URLSessionWebSocketTask
    - JavaScript: Use native WebSocket API
    - See CLIENT_DOCUMENTATION at bottom for detailed specs
"""

import asyncio
import json
import struct
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Optional, Callable, Set
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import websockets library
try:
    import websockets
    from websockets import serve
    from websockets.asyncio.server import ServerConnection
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    ServerConnection = None
    logger.warning("websockets library not installed. Install with: pip install websockets")


# ============================================================================
# Protocol Constants
# ============================================================================

# Message format types
FORMAT_SPZ = 0
FORMAT_PLY = 1

# Header structure: seq_id (4) + format_type (4) + data_length (4) = 12 bytes
HEADER_FORMAT = ">III"  # Big-endian: uint32, uint32, uint32
HEADER_SIZE = 12

# Protocol version for client compatibility checking
PROTOCOL_VERSION = 1


@dataclass
class ClientState:
    """State for a connected WebSocket client."""
    websocket: 'ServerConnection'
    connected_at: float = field(default_factory=time.time)
    frames_sent: int = 0
    bytes_sent: int = 0
    last_seq_id: int = -1
    paused: bool = False
    client_info: dict = field(default_factory=dict)
    
    @property
    def connection_duration(self) -> float:
        return time.time() - self.connected_at


@dataclass
class StreamerStats:
    """Statistics for the WebSocket streamer."""
    total_frames_broadcast: int = 0
    total_bytes_broadcast: int = 0
    current_clients: int = 0
    peak_clients: int = 0
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "total_frames_broadcast": self.total_frames_broadcast,
            "total_bytes_broadcast": self.total_bytes_broadcast,
            "total_mb_broadcast": round(self.total_bytes_broadcast / (1024 * 1024), 2),
            "current_clients": self.current_clients,
            "peak_clients": self.peak_clients,
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


class WebSocketStreamer:
    """
    WebSocket server for streaming Gaussian splat data to clients.
    
    This class manages WebSocket connections and broadcasts frames to all
    connected clients. It runs asynchronously in a background thread.
    
    Example:
        streamer = WebSocketStreamer(host="0.0.0.0", port=8765)
        streamer.start()
        
        # Later, broadcast frames (thread-safe):
        streamer.broadcast_frame(seq_id=0, output_bytes=spz_data)
        
        # Or use as callback:
        config = ServerConfig(output_callback=streamer.broadcast_frame)
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        output_format: str = "spz",
        max_queue_size: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize WebSocket streamer.
        
        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on
            output_format: Expected output format ("spz" or "ply")
            max_queue_size: Max frames to buffer per client before dropping
            verbose: Enable verbose logging
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library required. Install with: pip install websockets"
            )
        
        self.host = host
        self.port = port
        self.output_format = output_format.lower()
        self.format_type = FORMAT_SPZ if self.output_format == "spz" else FORMAT_PLY
        self.max_queue_size = max_queue_size
        self.verbose = verbose
        
        # Client management
        self._clients: Set[ClientState] = set()
        self._clients_lock = threading.Lock()
        
        # Statistics
        self.stats = StreamerStats()
        
        # Async event loop (runs in background thread)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._running = False
        
        # Frame queue for thread-safe broadcasting
        self._frame_queue: deque = deque(maxlen=max_queue_size)
        self._frame_event: Optional[asyncio.Event] = None
        
    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            logger.warning("WebSocket streamer already running")
            return
            
        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="WebSocketStreamer",
            daemon=True,
        )
        self._thread.start()
        
        # Wait for server to be ready
        timeout = 5.0
        start = time.time()
        while self._loop is None and (time.time() - start) < timeout:
            time.sleep(0.01)
            
        if self._loop is None:
            raise RuntimeError("WebSocket server failed to start within timeout")
            
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return
            
        self._running = False
        
        # Wait for the background thread to finish
        # The _serve() loop will exit when _running becomes False
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("WebSocket server thread did not stop cleanly")
            
        logger.info("WebSocket server stopped")
        
    def broadcast_frame(self, seq_id: int, output_bytes: bytes) -> None:
        """
        Broadcast a frame to all connected clients.
        
        This method is thread-safe and can be called from any thread.
        It's designed to be used as the output_callback for ServerConfig.
        
        Args:
            seq_id: Sequence ID of the frame
            output_bytes: Gaussian splat data (SPZ or PLY bytes)
        """
        if not self._running or self._loop is None:
            return
            
        # Build the binary message with header
        header = struct.pack(
            HEADER_FORMAT,
            seq_id,
            self.format_type,
            len(output_bytes),
        )
        message = header + output_bytes
        
        # Schedule broadcast on the event loop
        asyncio.run_coroutine_threadsafe(
            self._async_broadcast(seq_id, message),
            self._loop
        )
        
    def get_stats(self) -> dict:
        """Get current streamer statistics."""
        with self._clients_lock:
            self.stats.current_clients = len(self._clients)
        return self.stats.to_dict()
        
    def get_client_count(self) -> int:
        """Get current number of connected clients."""
        with self._clients_lock:
            return len(self._clients)
            
    # ========================================================================
    # Internal Methods
    # ========================================================================
    
    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self._loop.close()
            self._loop = None
            
    async def _serve(self) -> None:
        """Main server coroutine."""
        server = await serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,  # No limit on message size
        )
        self._server = server
        logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
        
        try:
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            # Gracefully close the server
            server.close()
            await server.wait_closed()
                
    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a client connection."""
        client = ClientState(websocket=websocket)
        remote = websocket.remote_address
        
        with self._clients_lock:
            self._clients.add(client)
            self.stats.current_clients = len(self._clients)
            self.stats.peak_clients = max(self.stats.peak_clients, len(self._clients))
            
        if self.verbose:
            logger.info(f"Client connected: {remote} (total: {self.stats.current_clients})")
            
        # Send welcome message with protocol info
        welcome = {
            "type": "welcome",
            "protocol_version": PROTOCOL_VERSION,
            "format": self.output_format,
            "format_type": self.format_type,
            "header_size": HEADER_SIZE,
            "server_time": time.time(),
        }
        await websocket.send(json.dumps(welcome))
        
        try:
            async for message in websocket:
                await self._handle_client_message(client, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            with self._clients_lock:
                self._clients.discard(client)
                self.stats.current_clients = len(self._clients)
                
            if self.verbose:
                duration = client.connection_duration
                logger.info(
                    f"Client disconnected: {remote} "
                    f"(duration: {duration:.1f}s, frames: {client.frames_sent}, "
                    f"bytes: {client.bytes_sent / 1024 / 1024:.2f}MB)"
                )
                
    async def _handle_client_message(self, client: ClientState, message: str) -> None:
        """Handle a message from a client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            
            if msg_type == "ping":
                await client.websocket.send(json.dumps({
                    "type": "pong",
                    "server_time": time.time(),
                }))
                
            elif msg_type == "status":
                stats = self.get_stats()
                stats["client"] = {
                    "frames_received": client.frames_sent,
                    "bytes_received": client.bytes_sent,
                    "last_seq_id": client.last_seq_id,
                    "paused": client.paused,
                    "connection_duration": client.connection_duration,
                }
                await client.websocket.send(json.dumps({
                    "type": "status",
                    **stats,
                }))
                
            elif msg_type == "pause":
                client.paused = True
                await client.websocket.send(json.dumps({
                    "type": "paused",
                    "last_seq_id": client.last_seq_id,
                }))
                
            elif msg_type == "resume":
                client.paused = False
                await client.websocket.send(json.dumps({
                    "type": "resumed",
                }))
                
            elif msg_type == "client_info":
                # Client can send info about itself
                client.client_info = data.get("info", {})
                if self.verbose:
                    logger.info(f"Client info: {client.client_info}")
                    
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message[:100]}")
            
    async def _async_broadcast(self, seq_id: int, message: bytes) -> None:
        """Broadcast a message to all connected clients."""
        with self._clients_lock:
            clients = list(self._clients)
            
        if not clients:
            return
            
        # Update stats
        self.stats.total_frames_broadcast += 1
        self.stats.total_bytes_broadcast += len(message) * len(clients)
        
        # Send to all non-paused clients
        tasks = []
        for client in clients:
            if not client.paused:
                tasks.append(self._send_to_client(client, seq_id, message))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        if self.verbose and len(clients) > 0:
            size_kb = len(message) / 1024
            logger.info(f"[WS] Broadcast seq {seq_id} ({size_kb:.1f}KB) to {len(tasks)} clients")
            
    async def _send_to_client(
        self,
        client: ClientState,
        seq_id: int,
        message: bytes,
    ) -> None:
        """Send a message to a specific client."""
        try:
            await client.websocket.send(message)
            client.frames_sent += 1
            client.bytes_sent += len(message)
            client.last_seq_id = seq_id
        except Exception as e:
            logger.debug(f"Failed to send to client: {e}")
            
    async def _close_all_clients(self) -> None:
        """Close all client connections."""
        with self._clients_lock:
            clients = list(self._clients)
            
        for client in clients:
            try:
                await client.websocket.close(1001, "Server shutting down")
            except Exception:
                pass


# ============================================================================
# Standalone Server (for testing)
# ============================================================================

def run_test_server(
    file_path: str,
    host: str = "0.0.0.0",
    port: int = 8765,
    interval: float = 1.0,
    loop: bool = True,
):
    """
    Run a test WebSocket server that streams an SPZ or PLY file to clients.
    
    Args:
        file_path: Path to an SPZ or PLY file to stream
        host: Host to bind to
        port: Port to listen on
        interval: Seconds between broadcasts (default: 1.0)
        loop: If True, keep re-broadcasting the file; if False, send once and wait
    """
    from pathlib import Path
    
    # Validate file exists and determine format
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    if suffix == ".spz":
        output_format = "spz"
    elif suffix == ".ply":
        output_format = "ply"
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .spz or .ply")
    
    # Read file contents
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    file_size_mb = len(file_data) / (1024 * 1024)
    
    streamer = WebSocketStreamer(
        host=host,
        port=port,
        output_format=output_format,
        verbose=True,
    )
    streamer.start()
    
    print(f"\n{'='*60}")
    print(f"WebSocket Test Server")
    print(f"{'='*60}")
    print(f"  URL: ws://{host}:{port}")
    print(f"  File: {file_path}")
    print(f"  Format: {output_format.upper()}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Interval: {interval}s")
    print(f"  Loop: {loop}")
    print(f"{'='*60}")
    print(f"\nWaiting for clients to connect...")
    print("Press Ctrl+C to stop\n")
    
    seq_id = 0
    try:
        while True:
            # Only broadcast if there are connected clients
            client_count = streamer.get_client_count()
            if client_count > 0:
                streamer.broadcast_frame(seq_id, file_data)
                seq_id += 1
                
                if not loop:
                    print(f"Sent frame {seq_id - 1}, waiting for more clients...")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()


# ============================================================================
# Client Documentation
# ============================================================================

CLIENT_DOCUMENTATION = """
================================================================================
WebSocket Streaming Client Implementation Guide
================================================================================

Protocol Version: 1
Server Default Port: 8765

--------------------------------------------------------------------------------
CONNECTION
--------------------------------------------------------------------------------

Connect to: ws://<server_host>:8765

On connection, server sends a JSON welcome message:
{
    "type": "welcome",
    "protocol_version": 1,
    "format": "spz",           // or "ply"
    "format_type": 0,          // 0=SPZ, 1=PLY
    "header_size": 12,
    "server_time": 1701234567.123
}

--------------------------------------------------------------------------------
RECEIVING FRAMES (Binary Messages)
--------------------------------------------------------------------------------

Each frame is sent as a binary WebSocket message with the following structure:

    Bytes 0-3:   seq_id      (uint32, big-endian) - Frame sequence number
    Bytes 4-7:   format_type (uint32, big-endian) - 0=SPZ, 1=PLY
    Bytes 8-11:  data_length (uint32, big-endian) - Length of gaussian data
    Bytes 12+:   data        (bytes)              - Gaussian splat data (SPZ/PLY)

Example parsing (Swift):
    
    func parseFrame(_ data: Data) -> (seqId: UInt32, formatType: UInt32, gaussianData: Data)? {
        guard data.count >= 12 else { return nil }
        
        let seqId = data[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let formatType = data[4..<8].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let dataLength = data[8..<12].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        
        guard data.count >= 12 + Int(dataLength) else { return nil }
        let gaussianData = data[12..<(12 + Int(dataLength))]
        
        return (seqId, formatType, Data(gaussianData))
    }

Example parsing (JavaScript):

    socket.onmessage = (event) => {
        const data = event.data;
        if (data instanceof Blob) {
            data.arrayBuffer().then(buffer => {
                const view = new DataView(buffer);
                const seqId = view.getUint32(0, false);      // big-endian
                const formatType = view.getUint32(4, false);
                const dataLength = view.getUint32(8, false);
                const gaussianData = buffer.slice(12, 12 + dataLength);
                
                console.log(`Received frame ${seqId}, ${dataLength} bytes`);
                // Process gaussianData (SPZ or PLY format)
            });
        }
    };

--------------------------------------------------------------------------------
SENDING CONTROL MESSAGES (JSON)
--------------------------------------------------------------------------------

Clients can send JSON messages to control streaming:

1. Ping (check connection):
   Send:    {"type": "ping"}
   Receive: {"type": "pong", "server_time": 1701234567.123}

2. Status (get server stats):
   Send:    {"type": "status"}
   Receive: {
       "type": "status",
       "total_frames_broadcast": 100,
       "total_bytes_broadcast": 104857600,
       "total_mb_broadcast": 100.0,
       "current_clients": 2,
       "peak_clients": 5,
       "uptime_seconds": 3600.0,
       "client": {
           "frames_received": 50,
           "bytes_received": 52428800,
           "last_seq_id": 49,
           "paused": false,
           "connection_duration": 120.5
       }
   }

3. Pause streaming:
   Send:    {"type": "pause"}
   Receive: {"type": "paused", "last_seq_id": 49}

4. Resume streaming:
   Send:    {"type": "resume"}
   Receive: {"type": "resumed"}

5. Client info (optional, for debugging):
   Send:    {"type": "client_info", "info": {"device": "Vision Pro", "app_version": "1.0"}}

--------------------------------------------------------------------------------
SWIFT CLIENT EXAMPLE (Mac / Vision Pro)
--------------------------------------------------------------------------------

import Foundation

class GaussianStreamClient {
    private var webSocketTask: URLSessionWebSocketTask?
    private let url: URL
    
    init(host: String, port: Int = 8765) {
        self.url = URL(string: "ws://\\(host):\\(port)")!
    }
    
    func connect() {
        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveMessage()
    }
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .data(let data):
                    self?.handleBinaryFrame(data)
                case .string(let text):
                    self?.handleTextMessage(text)
                @unknown default:
                    break
                }
                self?.receiveMessage()  // Continue receiving
                
            case .failure(let error):
                print("WebSocket error: \\(error)")
            }
        }
    }
    
    private func handleBinaryFrame(_ data: Data) {
        guard data.count >= 12 else { return }
        
        let seqId = data[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let formatType = data[4..<8].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let dataLength = data[8..<12].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        
        let gaussianData = data[12..<(12 + Int(dataLength))]
        
        print("Received frame \\(seqId): \\(dataLength) bytes (format: \\(formatType == 0 ? "SPZ" : "PLY"))")
        
        // TODO: Process gaussianData with your SPZ/PLY decoder
        // For SPZ, you'll need a decoder library
        // For PLY, you can parse the standard PLY format
    }
    
    private func handleTextMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }
        
        switch type {
        case "welcome":
            print("Connected! Format: \\(json["format"] ?? "unknown")")
        case "pong":
            print("Pong received")
        default:
            print("Received: \\(type)")
        }
    }
    
    func sendPing() {
        let message = URLSessionWebSocketTask.Message.string("{\\"type\\": \\"ping\\"}")
        webSocketTask?.send(message) { error in
            if let error = error { print("Send error: \\(error)") }
        }
    }
    
    func pause() {
        let message = URLSessionWebSocketTask.Message.string("{\\"type\\": \\"pause\\"}")
        webSocketTask?.send(message) { _ in }
    }
    
    func resume() {
        let message = URLSessionWebSocketTask.Message.string("{\\"type\\": \\"resume\\"}")
        webSocketTask?.send(message) { _ in }
    }
    
    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
    }
}

// Usage:
// let client = GaussianStreamClient(host: "192.168.1.100")
// client.connect()

--------------------------------------------------------------------------------
JAVASCRIPT CLIENT EXAMPLE (Web Browser)
--------------------------------------------------------------------------------

class GaussianStreamClient {
    constructor(host, port = 8765) {
        this.url = `ws://${host}:${port}`;
        this.socket = null;
        this.onFrame = null;  // Callback: (seqId, formatType, data) => {}
    }
    
    connect() {
        this.socket = new WebSocket(this.url);
        this.socket.binaryType = 'arraybuffer';
        
        this.socket.onopen = () => {
            console.log('Connected to gaussian stream server');
        };
        
        this.socket.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                this.handleBinaryFrame(event.data);
            } else {
                this.handleTextMessage(event.data);
            }
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.socket.onclose = () => {
            console.log('Disconnected from server');
        };
    }
    
    handleBinaryFrame(buffer) {
        const view = new DataView(buffer);
        const seqId = view.getUint32(0, false);
        const formatType = view.getUint32(4, false);
        const dataLength = view.getUint32(8, false);
        const gaussianData = buffer.slice(12, 12 + dataLength);
        
        console.log(`Frame ${seqId}: ${dataLength} bytes (${formatType === 0 ? 'SPZ' : 'PLY'})`);
        
        if (this.onFrame) {
            this.onFrame(seqId, formatType, gaussianData);
        }
    }
    
    handleTextMessage(text) {
        const data = JSON.parse(text);
        console.log('Server message:', data.type, data);
    }
    
    sendPing() {
        this.socket?.send(JSON.stringify({type: 'ping'}));
    }
    
    getStatus() {
        this.socket?.send(JSON.stringify({type: 'status'}));
    }
    
    pause() {
        this.socket?.send(JSON.stringify({type: 'pause'}));
    }
    
    resume() {
        this.socket?.send(JSON.stringify({type: 'resume'}));
    }
    
    disconnect() {
        this.socket?.close();
    }
}

// Usage:
// const client = new GaussianStreamClient('192.168.1.100');
// client.onFrame = (seqId, formatType, data) => {
//     // Process gaussian splat data
// };
// client.connect();

================================================================================
"""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WebSocket Streaming Server for Gaussian Splats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test server streaming an SPZ file
  python websocket_streamer.py --file gaussians.spz --port 8765
  
  # Run test server with custom interval
  python websocket_streamer.py --file output.ply --interval 0.5
  
  # Print client documentation
  python websocket_streamer.py --docs
"""
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--file", "-f", type=str, help="SPZ or PLY file to stream (required for test server)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between broadcasts (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true", help="Send file once per client instead of looping")
    parser.add_argument("--docs", action="store_true", help="Print client documentation")
    
    args = parser.parse_args()
    
    if args.docs:
        print(CLIENT_DOCUMENTATION)
    elif args.file:
        run_test_server(
            file_path=args.file,
            host=args.host,
            port=args.port,
            interval=args.interval,
            loop=not args.no_loop,
        )
    else:
        print("Usage:")
        print("  python websocket_streamer.py --file <spz_or_ply_file> [--port 8765]")
        print("  python websocket_streamer.py --docs")
        print("")
        print("For production, import WebSocketStreamer and integrate with server.py")

