"""Hermes media tools for the private Spark Telegram chat."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .workers import MediaWorkers

LOG = logging.getLogger("spark_media")
MEDIA_GUIDANCE = (
    "When the user asks what happens in an uploaded video or requests its timestamps, use Hermes's "
    "video_analyze tool first. Pass the Telegram attachment's saved path and the user's question. "
    "If deferred, search for the exact name video_analyze. Use elapsed video time and report an "
    "interval when a change falls between sampled frames. "
    "For a user-requested media export use the dedicated Spark media tools. "
    "Use create_textured_glb for a textured mesh, create_geometry_glb for geometry only, and "
    "generate_spark_video for a video. Use tool_search with the exact tool name if needed. "
    "These tools already have the installed models and handle Telegram delivery. "
    "Use Flash Next native images for OCR and photo questions and use the configured local "
    "text_to_speech tool if the user asks you to speak. Keep media work on the local tools."
)
CREATION = re.compile(r"\b(make|create|generate|turn|convert|build|render|export|produce)\b|生成|製作|制作|轉成|转成|變成|变成|匯出|导出", re.I)
THREE_D = re.compile(r"(?<![a-z0-9])(?:glb|mesh|3[ -]?d)(?![a-z0-9])|三維|三维|立體|立体", re.I)
VIDEO = re.compile(r"\b(video|mp4|clip)\b|影片|視頻|视频|短片", re.I)
GEOMETRY_ONLY = re.compile(r"\b(untextured|geometry.only|no textures?|without textures?)\b|純幾何|纯几何|不要貼圖|不要贴图", re.I)
DISCUSSION = re.compile(
    r"^\s*(?:please\s+)?(?:what|why|explain|describe|compare|is it)\b|"
    r"\b(?:how to|how do i|how can i|explain how|do not|don't|never|no need to)\b|"
    r"請解釋|请解释|怎樣|怎么|如何|不要|不要再",
    re.I,
)


def explicit_media_intent(text):
    if not isinstance(text, str):
        return None
    request = text.split("\n\n[", 1)[0].strip()
    if DISCUSSION.search(request) or not CREATION.search(request):
        return None
    is_3d, is_video = THREE_D.search(request), VIDEO.search(request)
    if is_3d and not is_video:
        kind = "create_geometry_glb" if GEOMETRY_ONLY.search(request) else "create_textured_glb"
        return kind, []
    if is_video and not is_3d and len(request) <= 1600:
        return "generate_spark_video", [request]
    return None
IMAGE_NAME = re.compile(r"img_[a-f0-9]{12}\.(?:jpe?g|png|webp)", re.IGNORECASE)

TOOLS = {
    "inspect_spark_photo": (
        "Use local Qwen3-VL as a fallback or second opinion for the Telegram owner's latest uploaded photo. "
        "For routine image understanding and OCR of an attached photo, use Flash Next's native image input first.",
        {"question": "What to inspect, read, or verify in the photo."},
    ),
    "create_textured_glb": (
        "Use native ComfyUI TRELLIS.2 to turn the owner's most recent Telegram photo into a textured 3D GLB "
        "with PBR maps. Call for a textured 3D asset, textured mesh, or GLB with materials. "
        "The tool sends the finished GLB to the same chat. Ask for a photo only if none is available.",
        {},
    ),
    "create_geometry_glb": (
        "Use TripoSG to turn the owner's latest Telegram photo into a geometry-only 3D GLB. "
        "Call when the user asks for an untextured mesh or geometry only. The tool sends the result to the chat.",
        {},
    ),
    "generate_spark_video": (
        "Use the local LTX 2.3 workflow to generate and send a short five-second MP4 with audio from a text description. "
        "Call when the user explicitly requests generating a video. This tool produces video and does not analyze clips.",
        {"prompt": "Visual and audio description of the requested clip."},
    ),
}


@dataclass(frozen=True)
class TelegramTurn:
    adapter: object
    loop: asyncio.AbstractEventLoop
    message_id: str


async def send_artifact(turn, owner, path):
    async def deliver():
        if path.suffix == ".mp4":
            return await turn.adapter.send_video(owner, str(path), reply_to=turn.message_id)
        return await turn.adapter.send_document(owner, str(path), file_name=path.name, reply_to=turn.message_id)
    if asyncio.get_running_loop() is turn.loop:
        return await deliver()
    future = asyncio.run_coroutine_threadsafe(deliver(), turn.loop)
    return await asyncio.wrap_future(future)


class Inbox:
    def __init__(self, directory):
        self.record = directory / "latest-photo.json"
        self.lock = Lock()

    @staticmethod
    def trusted(path, cache):
        return (path.is_absolute() and path.parent.resolve() == cache.resolve()
                and IMAGE_NAME.fullmatch(path.name) and path.is_file() and not path.is_symlink()
                and 0 < path.stat().st_size <= 12 * 1024**2)

    def latest(self, cache):
        with self.lock:
            try:
                value = Path(json.loads(self.record.read_text())["file"])
                return value if self.trusted(value, cache) else None
            except (OSError, KeyError, TypeError, ValueError):
                return None

    def remember(self, message_id, path, cache):
        if not self.trusted(path, cache):
            return
        with self.lock:
            try:
                stored = int(json.loads(self.record.read_text())["message"])
            except (OSError, KeyError, ValueError, TypeError):
                stored = -1
            if message_id < stored:
                return
            fd = self.record.with_suffix(".pending")
            fd.write_text(json.dumps({"message": message_id, "file": str(path)}))
            fd.chmod(0o600)
            fd.replace(self.record)


def register(ctx):
    config = json.loads((Path(__file__).parent / "media.json").read_text())
    root = Path(config["workbench"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Invalid Spark media workbench")
    owner = str(int(config["owner"]))
    workers = MediaWorkers(root)
    inbox = Inbox(workers.directory)
    latest_turn = None
    jobs = set()

    async def observe(event, gateway, **_kwargs):
        nonlocal latest_turn
        source = event.source
        if (getattr(source.platform, "value", None) != "telegram" or source.chat_type != "dm" or
                str(source.chat_id) != owner or str(source.user_id) != owner or source.is_bot or event.internal):
            return
        adapter = gateway.adapters.get(source.platform)
        message = str(event.message_id or "")
        if adapter is None or not message.isdecimal():
            return
        latest_turn = TelegramTurn(adapter, asyncio.get_running_loop(), message)
        event.channel_prompt = "\n\n".join(x for x in [event.channel_prompt, MEDIA_GUIDANCE] if x)
        for url, media in zip(event.media_urls, event.media_types):
            if isinstance(url, str) and isinstance(media, str) and media.startswith("image/"):
                from gateway.platforms.base import get_image_cache_dir
                await asyncio.to_thread(inbox.remember, int(message), Path(url), get_image_cache_dir())
                break

    async def execute(name, options, turn):
        try:
            if name == "generate_spark_video":
                path = await asyncio.to_thread(workers.comfy, "video", None, options[0])
            else:
                from gateway.platforms.base import get_image_cache_dir
                source = await asyncio.to_thread(inbox.latest, get_image_cache_dir())
                if source is None:
                    return {"error": "Please send a photo in this chat first."}
                if name == "inspect_spark_photo":
                    text = await asyncio.to_thread(workers.inspect, source, options[0])
                    return {"answer": text, "model": "Qwen3-VL 8B"}
                if name == "create_geometry_glb":
                    path = await asyncio.to_thread(workers.geometry, source)
                else:
                    path = await asyncio.to_thread(workers.comfy, "textured", source, None)
        except Exception as error:
            LOG.warning("%s failed: %s", name, type(error).__name__, exc_info=True)
            message = str(error) if isinstance(error, workers.user_error) else "The local worker failed. Check the Spark media log before retrying."
            return {"error": message}
        try:
            delivered = await send_artifact(turn, owner, path)
            if not delivered.success:
                raise RuntimeError("Telegram rejected the attachment")
        except Exception:
            LOG.warning("%s artifact delivery failed", name, exc_info=True)
            return {"error": "The artifact was saved on the Spark, but Telegram delivery failed.",
                    "file": str(path)}
        LOG.info("%s delivered %s (%d bytes) as Telegram message %s",
                 name, path.name, path.stat().st_size, getattr(delivered, "message_id", None))
        return {"success": True, "file": path.name, "bytes": path.stat().st_size, "delivered": True}

    async def finish_job(name, options, turn):
        result = await execute(name, options, turn)
        if result.get("error"):
            await turn.adapter.send(owner, result["error"], reply_to=turn.message_id)

    async def route_admitted(platform, source, text, message_id, **_kwargs):
        turn = latest_turn
        if (platform != "telegram" or source.get("chat_type") != "dm" or
                str(source.get("chat_id")) != owner or str(source.get("user_id")) != owner or
                turn is None or turn.message_id != str(message_id)):
            return
        intention = explicit_media_intent(text)
        if intention is None:
            return
        name, options = intention
        if name != "generate_spark_video":
            from gateway.platforms.base import get_image_cache_dir
            if await asyncio.to_thread(inbox.latest, get_image_cache_dir()) is None:
                return {"action": "handled", "reply": "Please send a photo for the 3D model."}
        if jobs or workers.gpu.locked():
            return {"action": "handled", "reply": "A local media job is running. Try again after it finishes."}
        LOG.info("Automatically routed a private Telegram request to %s",name)
        job = asyncio.create_task(finish_job(name, options, turn))
        jobs.add(job)
        def finished(task):
            jobs.discard(task)
            if not task.cancelled() and task.exception():
                LOG.error("Media job completion failed", exc_info=task.exception())
        job.add_done_callback(finished)
        titles = {"create_textured_glb": "TRELLIS.2 is generating the textured GLB.",
                  "create_geometry_glb": "TripoSG is generating the geometry GLB.",
                  "generate_spark_video": "LTX 2.3 is generating the video."}
        return {"action": "handled", "reply": titles[name]}

    def make_handler(name, arguments):
        async def call(args, **_kwargs):
            turn = latest_turn
            if turn is None or turn.loop.is_closed():
                return json.dumps({"error": "This tool is available in the private Spark Telegram chat."})
            from gateway.session_context import get_session_env
            if (get_session_env("HERMES_SESSION_PLATFORM") != "telegram" or
                    get_session_env("HERMES_SESSION_CHAT_ID") != owner or
                    get_session_env("HERMES_SESSION_USER_ID") != owner):
                return json.dumps({"error": "This tool is available in the private Spark Telegram chat."})
            reply_to = get_session_env("HERMES_SESSION_MESSAGE_ID")
            if reply_to and reply_to.isdecimal():
                turn = TelegramTurn(turn.adapter, turn.loop, reply_to)
            if not isinstance(args, dict) or set(args) != set(arguments):
                return json.dumps({"error": "Invalid media arguments."})
            options = [args[k] for k in arguments]
            if any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 1600 for v in options):
                return json.dumps({"error": "Provide a description between 1 and 1600 characters."})
            return json.dumps(await execute(name, options, turn))
        return call

    ctx.register_hook("pre_gateway_dispatch", observe)
    ctx.register_hook("post_gateway_admission", route_admitted)
    for name, (description, properties) in TOOLS.items():
        schema = {"name": name, "description": description, "parameters": {
            "type": "object", "properties": {key: {"type": "string", "description": value}
                                              for key, value in properties.items()},
            "required": list(properties), "additionalProperties": False,
        }}
        ctx.register_tool(name=name, toolset="spark_media", schema=schema, handler=make_handler(name, properties),
                          is_async=True, emoji="📦")
