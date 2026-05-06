## System Architecture

```mermaid
graph LR
    User[Client] -->|POST /ask| API[FastAPI Service]

    subgraph Inference
        API -->|embed query| Embed[BGE-small<br/>sentence-transformers]
        API -->|vector search| Qdrant[(Qdrant<br/>2540 chunks)]
        API -->|chat completion| LLM{LLM Backend}
        LLM -->|dev| Ollama[Ollama<br/>Qwen 2.5 7B]
        LLM -->|prod| vLLM[vllm-metal / vLLM<br/>Qwen 2.5 7B 4bit]
    end

    subgraph Observability
        API -.metrics.-> Prom[Prometheus]
        API -.traces.-> LF[LangFuse]
        Prom --> Graf[Grafana]
    end

    subgraph Eval
        RunEval[evaluation/run_eval.py] -->|Ragas metrics| MLflow[(MLflow)]
        RunEval -.uses.-> API
    end

    classDef storage fill:#e8d5ff,stroke:#5a3e8a
    classDef service fill:#d4e8ff,stroke:#3e5a8a
    classDef obs fill:#d5ffe8,stroke:#3e8a5a
    class Qdrant,MLflow storage
    class API,Embed,Ollama,vLLM,LLM service
    class Prom,LF,Graf,RunEval obs
```

The API is the only stateful service. Qdrant holds chunk embeddings; MLflow holds eval runs. Ollama / vLLM are stateless inference servers swapped via `INFERENCE_BACKEND` env var. Observability is fully additive — the system runs unchanged without LangFuse keys or with Prometheus disabled.