"""Download GGUF weights to the standard HF cache without loading a model."""
from __future__ import annotations

import os
import re
import sys


def select_files(files: list[str], quant: str) -> list[str]:
    ggufs = sorted(f for f in files if f.lower().endswith('.gguf') and 'mmproj' not in f.lower())
    matching = [f for f in ggufs if (quant or 'q4_k_m').lower() in f.lower()]
    if matching:
        return matching
    if quant:
        raise ValueError(f'No GGUF files match quantization {quant!r}.')
    # Single-format repos such as GPT-OSS use MXFP4 rather than Q4_K_M.
    families = {re.sub(r'-\d+-of-\d+(?=\.gguf$)', '', f, flags=re.I) for f in ggufs}
    if len(families) == 1:
        return ggufs
    raise ValueError('No unambiguous default GGUF. Specify org/repo:QUANT to choose a quantization.')


def main(argv: list[str]) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print('Install download support: python3 -m pip install -r requirements-models.txt', file=sys.stderr)
        return 1
    repo, quant = argv
    token = os.environ.get('HF_TOKEN') or None
    try:
        api = HfApi(endpoint=os.environ.get('HF_ENDPOINT') or None, token=token)
        info = api.model_info(repo, revision='main')
        # Pin every shard to the same revision, even if main changes during download.
        revision = info.sha
        files = select_files(api.list_repo_files(repo, revision=revision), quant)
        for filename in files:
            print(f'Downloading {filename}', flush=True)
            path = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                   endpoint=api.endpoint, token=token)
            print(f'Cached: {path}', flush=True)
    except Exception as exc:
        # Do not echo authenticated URLs or credentials from transport exceptions.
        print(f'Download failed ({type(exc).__name__}). Check repository access, quantization, and disk space.',
              file=sys.stderr)
        return 1
    print('Download complete. The running model and llama service were not changed.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
