import json
import multiprocessing as mp
import os
import time
import traceback


EVAL_FILE = "/home/azureuser/cloudfiles/code/Users/edcastr/TrainingSLM/EvaluationDataset9.jsonl"
OUTPUT_FILE = "/home/azureuser/cloudfiles/code/Users/edcastr/TrainingSLM/consolidated_answers/consolidated_MiniCPM-1B-sft-bf16.jsonl"

FT_MODEL = "Edcastro/MiniCPM-1B-sft-bf16-edcastr_JavaScript-v1"
OOB_MODEL = "openbmb/MiniCPM-1B-sft-bf16"

MAX_NEW_TOKENS = 200
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))


def _is_transport_disconnected(err: OSError) -> bool:
    return err.errno == 107 or "Transport endpoint is not connected" in str(err)


def resolve_writable_output_path(preferred_path: str) -> str:
    out_dir = os.path.dirname(preferred_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
        return preferred_path
    except OSError as err:
        if not _is_transport_disconnected(err):
            raise

        fallback_dir = os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/TrainingSLM_outputs")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, os.path.basename(preferred_path))
        print(
            "Warning: output path mount is disconnected; "
            f"writing results to fallback path: {fallback_path}"
        )
        return fallback_path


def build_prompt(question: str, tokenizer) -> str:
    # Prefer tokenizer chat template so each model gets its native input format.
    try:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback to the MiniCPM training-style template.
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def load_questions_and_answers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows


def enable_dynamic_cache_compat():
    """Patch transformers DynamicCache for MiniCPM implementations expecting seen_tokens."""
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "seen_tokens"):
            if hasattr(DynamicCache, "get_seq_length"):
                DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
            else:
                # Conservative fallback for older/newer cache variants.
                DynamicCache.seen_tokens = property(
                    lambda self: getattr(self, "_seen_tokens", 0)
                )
            print("Applied DynamicCache compatibility patch (seen_tokens)", flush=True)
    except Exception:
        # Non-fatal: generation fallback handles incompatible cache paths.
        pass


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id}")
    enable_dynamic_cache_compat()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    # Some model configs include sampling fields even when scoring should be greedy.
    # Clear these to avoid noisy warnings with do_sample=False.
    try:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    except Exception:
        pass

    model.eval()
    print(f"Loaded: {model_id}", flush=True)
    return model, tokenizer


def generate_answers_batch(model, tokenizer, questions, batch_size: int, role: str):
    import torch

    answers = []
    total = len(questions)
    use_cache_enabled = True
    for start in range(0, total, batch_size):
        batch_t0 = time.time()
        batch_questions = questions[start : start + batch_size]
        prompts = [build_prompt(q, tokenizer) for q in batch_questions]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=use_cache_enabled,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except AttributeError as err:
                # Newer transformers cache objects can be incompatible with some custom
                # MiniCPM prepare_inputs_for_generation implementations.
                if "seen_tokens" in str(err) or "DynamicCache" in str(err):
                    if use_cache_enabled:
                        print(
                            f"[{role}] Warning: cache incompatibility detected; "
                            "retrying with use_cache=False for this model.",
                            flush=True,
                        )
                    use_cache_enabled = False
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    raise

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i][input_lengths[i] :]
            answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        print(
            f"[{role}] Batch done: {min(start + batch_size, total)}/{total} "
            f"in {time.time() - batch_t0:.2f}s (use_cache={use_cache_enabled})",
            flush=True,
        )

    return answers


def run_model_inference(role: str, model_id: str, questions, batch_size: int):
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = load_model(model_id)
    answers = generate_answers_batch(
        model, tokenizer, questions, batch_size=batch_size, role=role
    )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answers


def worker(role: str, model_id: str, questions, batch_size: int, gpu_ids: str, queue):
    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        os.environ["PYTHONUNBUFFERED"] = "1"

        t0 = time.time()
        answers = run_model_inference(role, model_id, questions, batch_size=batch_size)
        queue.put(
            {
                "role": role,
                "answers": answers,
                "elapsed_s": time.time() - t0,
                "error": None,
            }
        )
    except Exception:
        queue.put(
            {
                "role": role,
                "answers": None,
                "elapsed_s": 0,
                "error": traceback.format_exc(),
            }
        )


def main():
    import torch

    os.environ["HF_HUB_CACHE"] = "/mnt/cache"
    os.environ["HF_DATASETS_CACHE"] = "/mnt/cache"
    os.environ["HF_HOME"] = "/mnt/cache"

    rows = load_questions_and_answers(EVAL_FILE)
    print(f"Loaded {len(rows)} evaluation rows")
    questions = [
        row["Question"].encode("utf-8", errors="replace").decode("utf-8")
        for row in rows
    ]

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected GPUs: {gpu_count}")
    print(f"Batch size: {BATCH_SIZE}")

    ft_answers = None
    oob_answers = None

    # Run FT and OOB concurrently when at least 2 GPUs are available.
    if gpu_count >= 2:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        if gpu_count >= 4:
            ft_gpu_ids = "0,1"
            oob_gpu_ids = "2,3"
        else:
            ft_gpu_ids = "0"
            oob_gpu_ids = "1"

        print(f"Launching FT on GPUs: {ft_gpu_ids}")
        print(f"Launching OOB on GPUs: {oob_gpu_ids}", flush=True)

        ft_proc = ctx.Process(
            target=worker,
            args=("FT", FT_MODEL, questions, BATCH_SIZE, ft_gpu_ids, queue),
        )
        oob_proc = ctx.Process(
            target=worker,
            args=("OOB", OOB_MODEL, questions, BATCH_SIZE, oob_gpu_ids, queue),
        )

        ft_proc.start()
        oob_proc.start()

        results = {}
        for _ in range(2):
            result = queue.get()
            results[result["role"]] = result

        ft_proc.join()
        oob_proc.join()

        if results["FT"]["error"]:
            raise RuntimeError(f"FT worker failed:\n{results['FT']['error']}")
        if results["OOB"]["error"]:
            raise RuntimeError(f"OOB worker failed:\n{results['OOB']['error']}")

        ft_answers = results["FT"]["answers"]
        oob_answers = results["OOB"]["answers"]
        print(f"FT elapsed: {results['FT']['elapsed_s']:.2f}s")
        print(f"OOB elapsed: {results['OOB']['elapsed_s']:.2f}s")
    else:
        print("Fewer than 2 GPUs detected; running sequentially.", flush=True)
        ft_answers = run_model_inference(
            "FT", FT_MODEL, questions, batch_size=BATCH_SIZE
        )
        oob_answers = run_model_inference(
            "OOB", OOB_MODEL, questions, batch_size=BATCH_SIZE
        )

    if len(ft_answers) != len(rows) or len(oob_answers) != len(rows):
        raise RuntimeError(
            "Generated answer counts do not match evaluation rows: "
            f"rows={len(rows)}, ft={len(ft_answers)}, oob={len(oob_answers)}"
        )

    output_file = resolve_writable_output_path(OUTPUT_FILE)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for row, ft, oob in zip(rows, ft_answers, oob_answers):
                out_row = {
                    "Question": row["Question"],
                    "answer": row.get("Answer", ""),
                    "FT": ft,
                    "OOB": oob,
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    except OSError as err:
        if not _is_transport_disconnected(err):
            raise

        fallback_file = os.path.join(
            os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/TrainingSLM_outputs"),
            os.path.basename(OUTPUT_FILE),
        )
        os.makedirs(os.path.dirname(fallback_file), exist_ok=True)
        print(
            "Warning: output write failed due to disconnected mount; "
            f"retrying at fallback path: {fallback_file}"
        )
        with open(fallback_file, "w", encoding="utf-8") as f:
            for row, ft, oob in zip(rows, ft_answers, oob_answers):
                out_row = {
                    "Question": row["Question"],
                    "answer": row.get("Answer", ""),
                    "FT": ft,
                    "OOB": oob,
                }
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        output_file = fallback_file

    print(f"Saved: {output_file}", flush=True)


if __name__ == "__main__":
    main()