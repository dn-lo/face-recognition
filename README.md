# Face Recognition
Deep learning models for face detection and recognition.

# Requirements

This repository is tailored for developers with limited computational resources. We will focus on state-of-the-art models that comply with the following requirements:

- Open source, with both PyTorch code and pre-trained weights available.
- Pre-trained on extensive datasets to support transfer learning and fine-tuning on downstream tasks.
- The pipeline may consists of one or two stages (i.e. separating detection and recognition) depending on which approach leads to a better performance.
- Fine-tuning should be feasible on mid-range GPU (e.g., NVIDIA RTX A3000)
- Inference should be possible on devices without dedicated GPUs (e.g., MacBook Air M1). This may be achieved with proper optimization techniques such as model quantization and pruning.
