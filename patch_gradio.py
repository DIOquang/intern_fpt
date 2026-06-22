"""
Patch gradio/oauth.py to handle huggingface_hub >= 1.0
where HfFolder was removed.
Run: python patch_gradio.py
"""
import pathlib
import importlib.util

spec = importlib.util.find_spec('gradio')
if not spec:
    print('gradio not found, skipping patch')
    exit(0)

oauth_path = pathlib.Path(spec.origin).parent / 'oauth.py'
if not oauth_path.exists():
    print(f'oauth.py not found at {oauth_path}, skipping')
    exit(0)

src = oauth_path.read_text()

old = 'from huggingface_hub import HfFolder, whoami'
new = '''try:
    from huggingface_hub import HfFolder, whoami
except ImportError:
    from huggingface_hub import whoami
    class HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass'''

if old in src:
    oauth_path.write_text(src.replace(old, new))
    print(f'  [OK] Patched {oauth_path}')
elif 'except ImportError' in src:
    print('  [SKIP] Already patched')
else:
    print(f'  [WARN] Pattern not found in {oauth_path}')

# Also patch gradio_client/utils.py for Python 3.13 bool schema issue
try:
    import gradio_client.utils as gcu
    utils_path = pathlib.Path(gcu.__file__)
    src2 = utils_path.read_text()

    old2 = 'def get_type(schema: dict):\n    if "const" in schema:'
    new2 = 'def get_type(schema: dict):\n    if isinstance(schema, bool):\n        return "Any"\n    if "const" in schema:'
    if old2 in src2:
        utils_path.write_text(src2.replace(old2, new2))
        print(f'  [OK] Patched gradio_client/utils.py (get_type)')
except Exception as e:
    print(f'  [SKIP] gradio_client patch: {e}')
