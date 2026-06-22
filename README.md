---
title: RealESRGAN Pytorch
emoji: 🔥📹
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: User Friendly Image & Video Upscaler!
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## CodeFormer local inference setup

1. Run the app:

```bash
python app.py
```

2. Open tab **Nhận diện & phục hồi gương mặt**.
3. Click button **0. Cài CodeFormer local**.
4. The setup step will:
	- Clone CodeFormer source to `third_party/CodeFormer`
	- Install CodeFormer dependencies
	- Download model to `weights/codeformer.pth`
	- Copy model into `third_party/CodeFormer/weights/CodeFormer/codeformer.pth`

After setup, the **Phục hồi khuôn mặt (CodeFormer)** flow runs fully local and does not call remote CodeFormer APIs.