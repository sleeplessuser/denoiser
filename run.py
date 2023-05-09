from torch import cuda

from web.app import run_app
from ml.utils import read_config
from ml.pipeline import get_pipeline

config = read_config('web/config.yaml')
device = 'cuda' if cuda.is_available() else 'cpu'
pipeline = get_pipeline(config['ckpt'], device)

run_app(pipeline, config['queue_size'])
