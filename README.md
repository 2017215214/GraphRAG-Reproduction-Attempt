## 环境准备

1. **Python 环境**
   - 推荐 Python 3.10 及以上
   - 建议使用虚拟环境（如 conda 或 venv）

2. **依赖安装**
   ```bash
   git clone <本项目地址>
   cd Build_GraphRAG
   conda install -r conda_enviroments.txt
   ```
3. **大模型vLLM**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8001
   ```
4. **quick start**
   ```bash
   python your_director/quick_start.py
   ```