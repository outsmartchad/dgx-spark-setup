"""TripoSG worker for a verified upload staged by the media gateway."""

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

source, output = map(Path, sys.argv[1:])
pipeline = TripoSGPipeline.from_pretrained("/work/models/triposg", torch_dtype=torch.float16, local_files_only=True)
pipeline.to("cuda")
with Image.open(source) as image:
    foreground = image.convert("RGB")
with torch.inference_mode():
    vertices, faces = pipeline(image=foreground, num_inference_steps=20, use_flash_decoder=False,
                               dense_octree_depth=6, hierarchical_octree_depth=7,
                               generator=torch.Generator(device="cuda").manual_seed(42)).samples[0]
assert np.isfinite(vertices).all() and len(faces)>100
trimesh.Trimesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces)).export(output)
