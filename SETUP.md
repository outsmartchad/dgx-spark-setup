# Runtime layout and checks

The deployed Hermes checkout is `~/.local/share/spark-telegram-hermes/source`, pinned to `v0.23.3`. Its Python environment provides `.venv/bin/hermes`. The Telegram profile lives beside it under `telegram-profile`. Keep the Telegram token and allowed user setting in that private profile using Hermes's Telegram setup.

1. Prepare and launch the TensorFold recipe on the Spark. Use the Flash Next checkpoint and image recorded in the README, bind the API to `127.0.0.1:18889`, and start with `VIDEO=1`. The recipe's upstream defaults can use a different bind address, so verify the listening socket before use.
2. Start Ollama on `127.0.0.1:11434` with `qwen3-vl:8b` cached for the photo fallback.
3. Prepare the local media workbench at `~/.local/share/spark-multimodal`. Copy the two JSON files from `workflows/` into the workbench root. The plugin expects ComfyUI under `src/ComfyUI`, tested at commit `1994420682258539f9ea1d22a9857a14df6acd32`. The `.comfy` and `.tripo` Python environments run inside the GPU container. TripoSG reads `models/triposg` and runs `gateway_media_geometry.py`, copied from the plugin's `geometry.py`. Use the filenames declared in the workflow loaders when staging the checkpoints.
4. Install `spark-media` under `telegram-profile/plugins/`. Copy `media.example.json` to `media.json` there and fill in the private owner ID and absolute workbench path. Copy `hermes-native.yaml` to the profile's `config.yaml` when preparing a new profile. For an existing profile, merge the settings so its private provider configuration is preserved.
5. For local voice, install Faster Whisper, PyAV, NumPy, CPU PyTorch, and Kokoro in the workbench's `.voice` environment. Prime the workbench's Hugging Face cache with Faster Whisper Small and Kokoro-82M. Configure named Hermes STT and TTS command providers invoking `.voice/bin/python` with the plugin's `speech.py`, then `stt INPUT OUTPUT` or `tts INPUT OUTPUT`. Use absolute paths and Hermes's provider placeholders. The runner uses the offline cache at inference time.
6. Set up the profile's `workspace` and `home` directories. Install the included gateway unit under `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, and start `spark-hermes-telegram.service`.

Upstream references for [TensorFold on one Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), [Hermes Telegram setup](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/telegram), and [speech command providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/tts).

## Smoke checks

In a Spark shell reached over Tailscale SSH:

```sh
systemctl --user is-active spark-hermes-telegram.service
ss -ltn | awk '$4 ~ /:(18889|11434)$/ {print $4}'
curl -fsS http://127.0.0.1:18889/v1/models
```

In the private Telegram chat, send a normal text message, attach a photo and ask it to read a label, then send a voice note. For video, attach a short MP4 with known scene cuts and ask for their timestamps. Look for `video_analyze` in the tool progress. For TRELLIS.2, send a photo with a request for a textured GLB and verify the downloaded file opens with its textures.
