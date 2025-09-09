# Face Recognition

Deep learning models for face detection and recognition.

## Requirements

This repository is designed for developers with limited computational resources. We will focus on state-of-the-art models that comply with the following requirements:

- **Open source**: Both PyTorch code and pre-trained weights must be available.
- **Pre-trained on extensive datasets**: Enables transfer learning and fine-tuning on downstream tasks.
- **Pipeline design**: May consist of one or two stages (e.g., separating detection and recognition) depending on which approach delivers better performance.
- **Fine-tuning feasibility**: Can be trained on mid-range GPUs (e.g., NVIDIA RTX A3000)
- **Efficient inference**: Should run on devices without dedicated GPUs (e.g., MacBook Air M1), using proper optimization techniques such as model quantization and pruning.

---

## Development Container Setup

This repository includes a **Dev Container** configuration to simplify environment setup. Using a dev container ensures a consistent environment across different machines and avoids dependency issues.

### Step-by-Step Guide to Install the Container

1. **Install prerequisites**:
   - [Visual Studio Code](https://code.visualstudio.com/)
   - [Docker](https://www.docker.com/get-started)
   - VS Code extensions:
     - **Dev Containers** ([official guide](https://code.visualstudio.com/docs/devcontainers/containers))
     - **Python** (optional, for Python support)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/face-recognition.git
   cd face-recognition
   ```

3. **Open the repository in VS Code**:
   ```bash
   code .
   ```

4. **Reopen in Dev Container**:
   - GUI method: Press `F1` → type `Dev Containers: Reopen in Container`
   - **Alternative (Command Line)**: Run the following from the repo root:
     ```bash
     devcontainer reopen
     ```
   - VS Code will automatically build the container using the `.devcontainer/Dockerfile` and `.devcontainer/devcontainer.json`.

5. **Wait for the container to build**:
   - The first build may take a few minutes.
   - Dependencies like PyTorch and other Python packages will be installed automatically inside the container.

6. **Verify the environment**:
   - Open a terminal inside VS Code (`Ctrl + `) and run:
     ```bash
     python -c "import torch; print(torch.__version__)"
     ```
   - This confirms PyTorch is installed and accessible.

7. **Run your scripts**:
   - All commands run inside VS Code terminal are executed in the container.
   - You can now start training, fine-tuning, or testing models without worrying about local dependencies.

---

Using this setup ensures your development environment is fully reproducible and portable across machines, making it ideal for collaboration and testing.
