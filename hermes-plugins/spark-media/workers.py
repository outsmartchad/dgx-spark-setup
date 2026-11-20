"""Bounded local media workers. Model inputs never become shell arguments."""

import base64
import http.client
import io
import json
import logging
import os
import socket
import stat
import struct
import subprocess
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock

from PIL import Image

LOG = logging.getLogger("spark_media.worker")
IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3"
DOCKER = "/usr/bin/docker"
MEDIA_PORT = 18193
MAX_PHOTO = 12 * 1024**2
MAX_EXPORT = 45 * 1024**2


class MediaError(Exception):
    pass


def local_api(port, path, payload=None, timeout=15):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        conn.request("GET" if body is None else "POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        text = response.read(4 * 1024**2)
        if response.status != 200:
            raise RuntimeError(f"Loopback worker {port} returned HTTP {response.status}: {text[:600]!r}")
        return json.loads(text)
    finally:
        conn.close()


def normalize_photo(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= MAX_PHOTO:
            raise MediaError("The uploaded photo is too large or unreadable.")
        data = stream.read(MAX_PHOTO + 1)
    if len(data) > MAX_PHOTO:
        raise MediaError("The uploaded photo is too large.")
    with Image.open(io.BytesIO(data)) as original:
        if original.format not in {"JPEG", "PNG", "WEBP"} or original.width * original.height > 32_000_000:
            raise MediaError("Send a JPEG, PNG, or WebP photo under 32 megapixels.")
        original.thumbnail((1400, 1400))
        output = io.BytesIO()
        original.convert("RGB").save(output, format="PNG")
        return output.getvalue()


def verify_export(path, kind):
    size = path.stat().st_size
    if not 4096 < size <= MAX_EXPORT or path.is_symlink():
        raise MediaError("The export is empty or too large for Telegram.")
    with path.open("rb") as stream:
        header = stream.read(20)
        if kind == "video":
            if header[4:8] != b"ftyp":
                raise MediaError("The worker did not produce a valid MP4.")
            return
        magic, version, total, length, block = struct.unpack("<4sIII4s", header)
        if (magic, version, total, block) != (b"glTF", 2, size, b"JSON") or not 0 < length < size - 20:
            raise MediaError("The worker did not produce a valid GLB.")
        manifest = json.loads(stream.read(length))
    primitives = [p for mesh in manifest.get("meshes", ()) for p in mesh.get("primitives", ())]
    if not primitives:
        raise MediaError("The GLB contains no mesh.")
    if kind == "textured":
        materials = manifest.get("materials", ())
        textures = manifest.get("textures", ())
        images = manifest.get("images", ())
        for p in primitives:
            if "TEXCOORD_0" not in p.get("attributes", {}):
                raise MediaError("The GLB mesh is missing UV coordinates.")
            material = materials[p["material"]]
            pbr = material.get("pbrMetallicRoughness", {})
            refs = [pbr.get("baseColorTexture"), pbr.get("metallicRoughnessTexture"),
                    material.get("normalTexture"), material.get("occlusionTexture")]
            if any(not r or "bufferView" not in images[textures[r["index"]]["source"]] for r in refs):
                raise MediaError("The GLB is missing embedded material maps.")


class MediaWorkers:
    user_error = MediaError

    def __init__(self, root):
        self.root = root
        self.directory = root / "gateway-media"
        self.directory.mkdir(mode=0o700, exist_ok=True)
        self.gpu = Lock()

    @contextmanager
    def gpu_job(self):
        if not self.gpu.acquire(blocking=False):
            raise MediaError("A local media job is running. Retry after it finishes.")
        try:
            yield
        finally:
            self.gpu.release()

    def job_dir(self):
        job = self.directory / uuid.uuid4().hex
        job.mkdir(mode=0o700)
        return job

    def in_container(self, path):
        return "/work/" + str(path.relative_to(self.root))

    def docker(self, *args, timeout=30, check=True):
        return subprocess.run([DOCKER, *args], cwd=self.root, timeout=timeout, check=check,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True)

    def container_args(self, mem):
        return ("--rm", "--gpus", "all", "--memory", mem, "--memory-swap", mem,
                "--cpus", "8", "--shm-size", "768m", "--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/work", "-e", "XDG_CACHE_HOME=/work/.cache", "-e", "HF_HUB_OFFLINE=1",
                "-v", f"{self.root}:/work")

    def inspect(self, source, question):
        with self.gpu_job():
            encoded = base64.b64encode(normalize_photo(source)).decode()
            data = {"model": "qwen3-vl:8b", "stream": False, "keep_alive": 0,
                    "messages": [{"role": "user", "content": question, "images": [encoded]}],
                    "options": {"num_ctx": 4096, "num_predict": 512, "temperature": 0}}
            return local_api(11434, "/api/chat", data, timeout=140)["message"]["content"]

    def geometry(self, source):
        with self.gpu_job():
            job = self.job_dir()
            photo, artifact = job / "source.png", job / "spark-geometry.glb"
            photo.write_bytes(normalize_photo(source))
            result = self.docker("run", *self.container_args("15g"), "--network", "none",
                                 "-e", "PYTHONPATH=/work/src/TripoSG", "-w", "/work/src/TripoSG", IMAGE,
                                 "/work/.tripo/bin/python", "/work/gateway_media_geometry.py",
                                 self.in_container(photo), self.in_container(artifact),
                                 timeout=390, check=False)
            if result.returncode:
                LOG.warning("TripoSG exited %s: %s", result.returncode, result.stderr[-1400:])
                raise MediaError("TripoSG could not generate a mesh.")
            verify_export(artifact, "geometry")
            return artifact

    def load_workflow(self, kind, prompt):
        filename = "native-trellis-prompt.json" if kind == "textured" else "ltx23-smoke-prompt.json"
        graph = json.loads((self.root / filename).read_text())
        if kind == "textured":
            next(n for n in graph.values() if n["class_type"] == "LoadImage")["inputs"]["image"] = "source.png"
            next(n for n in graph.values() if n["class_type"] == "Save3DAdvanced")["inputs"]["filename_prefix"] = "textured"
        else:
            next(n for n in graph.values() if n["class_type"] == "CLIPTextEncode")["inputs"]["text"] = prompt
            next(n for n in graph.values() if n["class_type"] == "SaveVideo")["inputs"]["filename_prefix"] = "video"
        return graph

    def wait_for_comfy(self):
        for _ in range(40):
            with suppress(OSError, RuntimeError):
                local_api(MEDIA_PORT, "/system_stats", timeout=2)
                return
            time.sleep(2)
        raise MediaError("The local ComfyUI worker did not become ready.")

    def render(self, graph, deadline):
        receipt = local_api(MEDIA_PORT, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
        prompt_id = str(uuid.UUID(receipt["prompt_id"]))
        while time.monotonic() < deadline:
            result = local_api(MEDIA_PORT, "/history/" + prompt_id).get(prompt_id)
            if result:
                status = result["status"]
                if status["status_str"] != "success":
                    details = [(data.get("node_type"), data.get("exception_type"))
                               for event, data in status.get("messages", ()) if event == "execution_error"]
                    LOG.warning("ComfyUI rejected the media job: %s", details)
                    raise MediaError("ComfyUI could not complete the local workflow.")
                return
            time.sleep(3)
        raise MediaError("The local media job timed out.")

    def comfy(self, kind, source, prompt):
        with self.gpu_job():
            job = self.job_dir()
            output = job / "output"
            output.mkdir()
            if source is not None:
                (job / "source.png").write_bytes(normalize_photo(source))
            graph = self.load_workflow(kind, prompt)
            # Docker's loopback publication reserves the port or fails, even if another app won
            # the probe/start race. Comfy's listener lives inside this container.
            name = "spark-media-" + job.name[:12]
            container = self.docker("run", "-d", *self.container_args("22g"), "--name", name,
                                    "-p", f"127.0.0.1:{MEDIA_PORT}:8188", "-w", "/work/src/ComfyUI", IMAGE,
                                    "/work/.comfy/bin/python", "-u", "main.py", "--listen", "0.0.0.0",
                                    "--port", "8188", "--lowvram", "--reserve-vram", "11",
                                    "--disable-all-custom-nodes", "--offline",
                                    "--input-directory", self.in_container(job),
                                    "--output-directory", self.in_container(output)).stdout.strip()
            try:
                self.wait_for_comfy()
                self.render(graph, time.monotonic() + (440 if kind == "video" else 285))
                suffix = ".mp4" if kind == "video" else ".glb"
                files = list(output.rglob("*" + suffix))
                if len(files) != 1 or not files[0].resolve().is_relative_to(output):
                    raise MediaError("The worker did not return a single verified export.")
                verify_export(files[0], kind)
                destination = job / ("spark-video.mp4" if kind == "video" else "spark-textured.glb")
                files[0].replace(destination)
                return destination
            except Exception:
                with suppress(OSError, subprocess.SubprocessError):
                    log = self.docker("logs", "--tail", "25", container, timeout=8, check=False)
                    LOG.warning("ComfyUI worker log: %s", (log.stderr + log.stdout)[-3600:])
                raise
            finally:
                with suppress(OSError, subprocess.SubprocessError):
                    self.docker("stop", "--time", "4", container, timeout=14, check=False)
