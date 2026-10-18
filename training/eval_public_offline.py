"""
Runs eval_public.py for two BERT-based classifiers whose tokenizers contact the
Hugging Face Hub while loading, which fails on compute nodes without internet.
The call only checks for Mistral tags, so it is replaced with a stub.
"""
import sys, types
import huggingface_hub
huggingface_hub.model_info = lambda *a, **k: types.SimpleNamespace(tags=[])
import eval_public
for m in ["citizenlab/distilbert-base-multilingual-cased-toxicity",
          "Hate-speech-CNERG/dehatebert-mono-english"]:
    sys.argv = ["eval_public.py", "--model", m]
    try:
        eval_public.main()
    except Exception as e:
        print(f"PUBLIC_FAIL model={m} err={type(e).__name__}: {e}", flush=True)
