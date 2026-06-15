#!/usr/bin/env python3
import os
import socket
import struct
import sys

import cv2
import numpy as np
import torch


HLOC_ROOT = os.environ.get(
    "F_VIGS_HLOC_ROOT",
    "/home/jorge/ros2_thesis_ws/src/LoopSplat/thirdparty/Hierarchical-Localization",
)
HOST = os.environ.get("F_VIGS_NETVLAD_HOST", "0.0.0.0")
PORT = int(os.environ.get("F_VIGS_NETVLAD_PORT", "5000"))

if HLOC_ROOT not in sys.path:
    sys.path.append(HLOC_ROOT)

try:
    from hloc.extractors.netvlad import NetVLAD
except Exception as exc:
    print(f"[NetVLADServer] failed to import hloc NetVLAD: {exc}", flush=True)
    raise


CONF = {
    "output": "global-feats-netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": 1024},
}


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img.unsqueeze(0)


def recvall(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


print(f"[NetVLADServer] starting on {HOST}:{PORT}", flush=True)
print(f"[NetVLADServer] hloc_root={HLOC_ROOT}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    netvlad = NetVLAD(CONF).to(device).eval()
except Exception as exc:
    print(f"[NetVLADServer] failed to initialize NetVLAD model: {exc}", flush=True)
    raise

print(f"[NetVLADServer] model ready on {device}", flush=True)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

request_count = 0

while True:
    print("[NetVLADServer] waiting for client connection...", flush=True)
    conn, addr = server.accept()
    print(f"[NetVLADServer] client connected: {addr}", flush=True)

    with conn:
        while True:
            size_data = recvall(conn, 4)
            if not size_data:
                print("[NetVLADServer] client disconnected", flush=True)
                break

            size = struct.unpack("!I", size_data)[0]
            img_data = recvall(conn, size)
            if img_data is None:
                print("[NetVLADServer] failed to receive image payload", flush=True)
                break

            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("[NetVLADServer] cv2.imdecode failed", flush=True)
                break

            inp = preprocess(img).to(device)

            try:
                with torch.no_grad():
                    desc = netvlad({"image": inp})["global_descriptor"]
            except Exception as exc:
                print(f"[NetVLADServer] descriptor extraction failed: {exc}", flush=True)
                break

            desc = desc.detach().cpu().numpy().astype(np.float32).reshape(-1)
            desc_bytes = desc.tobytes()

            try:
                conn.sendall(struct.pack("!I", len(desc_bytes)))
                conn.sendall(desc_bytes)
            except Exception as exc:
                print(f"[NetVLADServer] failed to send descriptor: {exc}", flush=True)
                break

            request_count += 1
            print(
                f"[NetVLADServer] request={request_count} image={img.shape[1]}x{img.shape[0]} descriptor_dim={desc.size}",
                flush=True,
            )
