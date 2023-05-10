from torch import cuda

from web.app import run_app
from ml.utils import read_config
from ml.pipeline import get_pipeline


def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read() 

if __name__ == "__main__":
    config = read_config('web/config.yaml')

    device = 'cuda' if cuda.is_available() else 'cpu'
    pipeline = get_pipeline(config['ckpt'], device)

    js = read_file('web/static/script.js')
    css = read_file('web/static/style.css')

    run_app(pipeline, js, css, config['queue_size'])
