# musetalk

MuseTalk one-click launch script for [cloudcompute.ru](https://app.cloudcompute.ru).

Provisions a GPU instance with [MuseTalk](https://github.com/TMElyralab/MuseTalk) v1.5
(audio-driven lip-sync) and runs the bundled Gradio web UI on port 7860.

## Launch

Via the [CloudCompute dashboard](https://app.cloudcompute.ru/applications/musetalk) — one click.

## Verify on a live box

```bash
# Check all required model files are present
ls /workspace/MuseTalk/models/musetalkV15/unet.pth
ls /workspace/MuseTalk/models/musetalkV15/musetalk.json
ls /workspace/MuseTalk/models/sd-vae/config.json
ls /workspace/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin
ls /workspace/MuseTalk/models/whisper/pytorch_model.bin
ls /workspace/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth
ls /workspace/MuseTalk/models/syncnet/latentsync_syncnet.pt
ls /workspace/MuseTalk/models/face-parse-bisent/79999_iter.pth
ls /workspace/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth

# Tail the server log
tail -f /var/log/cc-musetalk.log

# CLI inference (v1.5) over SSH — runs from the repo dir with the provisioned venv
cd /workspace/MuseTalk
source .venv/bin/activate
sh inference.sh v1.5 normal
# Output: results/test/v15/<input_basename>/<input_basename>.mp4
```

## License

MIT
