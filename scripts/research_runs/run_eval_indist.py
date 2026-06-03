from src.config import get_config
from src.eval.evaluate import run_eval
run_eval(get_config(), use_training_data=True)
