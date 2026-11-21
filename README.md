<div align="center">
  <img src="laya_torch_logo.jpg" alt="Laya-Torch AI Logo" width="100%">
  
  <h1>Laya-Torch: The PyTorch Port of the Laya Autonomous Decision Agent</h1>

  <p>
    <strong>A blazingly fast, multi-platform, typed decision agent for AI autonomous systems—now running natively on Windows, Linux, and CUDA via PyTorch.</strong>
  </p>

  <p>
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#benchmarks">Benchmarks</a> •
    <a href="#why-pytorch">Why PyTorch?</a>
  </p>
</div>

---

## 🚀 About Laya-Torch

Built by [VisionQuantech](https://visionquantech.com) — AI agents, apps & tools for small businesses.

**Laya-Torch** is the official PyTorch port of the groundbreaking [Laya](https://github.com/Voltaic/laya-mlx) project. Originally restricted to Apple Silicon (via Apple's MLX framework), this project brings the same deterministic, typed decision-making engine to **Windows**, **Linux**, and any hardware supported by PyTorch (CUDA, ROCm, CPU).

Laya-Torch provides a robust, high-performance **System One** decision model, built on ModernBERT. It is designed to make rapid, typed micro-decisions (e.g., scoring, boolean classification, dynamic routing) for autonomous AI agents, ensuring predictability, safety, and scale.

*Authored by: **Shiva***

## ✨ Key Features

- 💻 **Cross-Platform**: Run natively on Windows, Linux, and macOS.
- ⚡ **CUDA-Accelerated**: Leverage your NVIDIA GPUs for high-throughput inference using PyTorch's `scaled_dot_product_attention`.
- 🧠 **ModernBERT Engine**: State-of-the-art transformer architecture optimized for rapid context evaluation.
- 🛡️ **Typed Decisions**: Get structured JSON outputs, `noul` (boolean), `score` (float), and `choice` classifications out of the box. No more parsing messy LLM text.
- 📦 **Zero-Shot Routing**: Built-in routing capabilities to automatically classify and delegate tasks.

## 📦 Installation

To get started with Laya-Torch, ensure you have Python 3.10+ installed. We recommend using [`uv`](https://github.com/astral-sh/uv) or `pip` in a virtual environment.

```bash
# Clone the repository
git clone https://github.com/shiva/laya-torch.git
cd laya-torch

# Install dependencies (including torch, transformers, and safetensors)
uv pip install -e .
```

## 🎮 Quick Start

Laya-Torch makes evaluating states and choosing the safest actions incredibly simple. Here's a basic example:

```python
from laya_mlx import Agent

# Load the agent (downloads weights automatically if not cached)
agent = Agent("aac6fef/laya-multilingual-mlx", dtype="float16")

state = "The user is asking for a refund because their server crashed."
questions = {
    "intent": {
        "type": "choice",
        "instructions": "Classify the user intent.",
        "criteria": ["refund_request", "technical_support", "complaint"]
    },
    "urgency": {
        "type": "score",
        "instructions": "Rate the urgency of this ticket."
    }
}

response = agent.predict(state, questions)

print(response["answers"]["intent"]["choice"]) # Output: "refund_request"
print(response["answers"]["urgency"]["score"]) # Output: 0.89
```

## 🔍 Why PyTorch?

The original Laya framework was built on MLX, which limited its reach to Apple Silicon users. By porting the entire model architecture, custom RoPE embeddings, and safe-tensor loading to PyTorch, we've:

1. **Democratized Access**: Windows users can now build and deploy Laya agents locally.
2. **Enterprise Readiness**: Deploy to standard cloud instances (AWS, GCP, Azure) equipped with NVIDIA GPUs.
3. **Ecosystem Integration**: Seamlessly integrate with the broader PyTorch ecosystem (`torch.compile`, Hugging Face, etc.).

## 🧪 Testing

The repository maintains the rigorous 140+ test suite from the original upstream, ensuring 100% parity and safety constraints.

```bash
uv run pytest tests/
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Generated and ported to PyTorch by Shiva to empower autonomous systems worldwide.*