# Face Recognition
Deep learning models for face detection and recognition.

# Requirements

This repository is designed for developers with limited computational resources. We will focus on state-of-the-art models that comply with the following requirements:

- **Open source**: Both PyTorch code and pre-trained weights must be available.
- **Pre-trained on extensive datasets**: Enables transfer learning and fine-tuning on downstream tasks.
- **Pipeline design**: May consists of one or two stages (e.g., separating detection and recognition) depending on which approach delivers better performance.
- **Fine-tuning feasibility**: Can be trained on mid-range GPUs (e.g., NVIDIA RTX A3000)
- **Efficient inference**: Should run on devices without dedicated GPUs (e.g., MacBook Air M1), using proper optimization techniques such as model quantization and pruning.
