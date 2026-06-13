"""Local, cloud-free inference path for Scholar Lens (llama.cpp / GGUF).

Why this exists
---------------
The hosted app uses Modal for convenience, but Scholar Lens is small enough to
run entirely on a single consumer GPU (or even CPU). This module proves it:
it runs the same summarize / synthesize prompts through a quantized GGUF model
with ``llama-cpp-python`` — no cloud APIs.

This unlocks the hackathon's **Off-Grid** (no cloud APIs) and **Llama Champion**
(llama.cpp runtime) merit badges, and it is the basis for the NVIDIA pitch:
"give me an RTX-class GPU and the whole thing is local, private, and free to run."

Setup
-----
    pip install llama-cpp-python      # add a CUDA wheel for GPU acceleration

Point it at a GGUF model (download once, e.g. a Qwen2.5 Instruct GGUF):
    set SCHOLAR_LENS_GGUF=C:\\models\\qwen2.5-3b-instruct-q4_k_m.gguf   # Windows
    export SCHOLAR_LENS_GGUF=/models/qwen2.5-3b-instruct-q4_k_m.gguf    # Unix

Usage
-----
    python local_inference.py --benchmark
    python local_inference.py --summarize "Long abstract text..."
    python local_inference.py --ask "Question?" --context "[1] Title: ... Abstract: ..."
"""

from __future__ import annotations

import argparse
import os
import time

MODEL_PATH = os.getenv("SCHOLAR_LENS_GGUF", "").strip()
# Number of layers to offload to GPU. -1 = all layers (full GPU). 0 = CPU only.
GPU_LAYERS = int(os.getenv("SCHOLAR_LENS_GPU_LAYERS", "-1"))
CONTEXT_WINDOW = int(os.getenv("SCHOLAR_LENS_CTX", "8192"))

_LLM = None


def _load_model():
    """Load the GGUF model once (lazy)."""
    global _LLM
    if _LLM is not None:
        return _LLM
    if not MODEL_PATH:
        raise SystemExit(
            "Set SCHOLAR_LENS_GGUF to a local GGUF model file first "
            "(see the module docstring)."
        )
    from llama_cpp import Llama

    _LLM = Llama(
        model_path=MODEL_PATH,
        n_ctx=CONTEXT_WINDOW,
        n_gpu_layers=GPU_LAYERS,
        verbose=False,
    )
    return _LLM


def _chat(prompt: str, max_tokens: int = 350) -> tuple[str, dict]:
    """Run one chat completion; return (text, timing stats)."""
    llm = _load_model()
    start = time.perf_counter()
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.15,
    )
    elapsed = time.perf_counter() - start
    text = result["choices"][0]["message"]["content"].strip()
    completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
    stats = {
        "seconds": round(elapsed, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(completion_tokens / elapsed, 1) if elapsed else 0.0,
    }
    return text, stats


def summarize(text: str) -> str:
    prompt = (
        "Summarize the following research paper context in 4-6 clear sentences. "
        "Cover the main contribution, methods, and key results/findings.\n\n"
        f"Paper context:\n{text}"
    )
    answer, _ = _chat(prompt, max_tokens=250)
    return answer


def synthesize(question: str, context: str) -> str:
    prompt = (
        "You are a meticulous research assistant. Using ONLY the numbered paper "
        "abstracts below, write a clear, synthesized answer to the question. "
        "Cite every claim with the matching source number in square brackets, "
        "e.g. [1] or [2][3]. Never invent sources.\n\n"
        f"{context}\n\nQuestion: {question}\n\nSynthesized answer (with [n] citations):"
    )
    answer, _ = _chat(prompt, max_tokens=450)
    return answer


def benchmark() -> None:
    """Print throughput so we can report consumer-GPU numbers in the README."""
    sample_context = (
        "[1] Title: Aerosol-cloud interactions in climate models\n"
        "Abstract: We review parameterizations of aerosol-cloud interactions and "
        "quantify the spread in radiative forcing estimates across CMIP6 models, "
        "finding the largest uncertainty in the cloud-lifetime effect.\n\n"
        "[2] Title: Machine learning for satellite precipitation\n"
        "Abstract: A convolutional model improves sub-daily precipitation retrieval "
        "skill over mountainous terrain relative to operational baselines."
    )
    question = "Where do papers disagree on aerosol-cloud interaction uncertainty?"
    answer, stats = _chat(
        "You are a research assistant. Using ONLY these abstracts, answer with "
        f"[n] citations.\n\n{sample_context}\n\nQuestion: {question}\n\nAnswer:",
        max_tokens=300,
    )
    print("=== Scholar Lens local benchmark ===")
    print(f"Model file        : {MODEL_PATH}")
    print(f"GPU layers        : {GPU_LAYERS} (-1 = all on GPU, 0 = CPU)")
    print(f"Context window    : {CONTEXT_WINDOW}")
    print(f"Completion tokens : {stats['completion_tokens']}")
    print(f"Latency (s)       : {stats['seconds']}")
    print(f"Throughput (tok/s): {stats['tokens_per_sec']}")
    print("\n--- sample answer ---")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scholar Lens local (llama.cpp) inference.")
    parser.add_argument("--benchmark", action="store_true", help="Run a throughput benchmark.")
    parser.add_argument("--summarize", metavar="TEXT", help="Summarize the given paper text.")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question (use with --context).")
    parser.add_argument("--context", metavar="CONTEXT", default="", help="Numbered abstracts for --ask.")
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    elif args.summarize:
        print(summarize(args.summarize))
    elif args.ask:
        if not args.context:
            raise SystemExit("--ask requires --context with numbered abstracts.")
        print(synthesize(args.ask, args.context))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
