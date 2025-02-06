"""TTS Controller package for managing multiple TTS models."""

__version__ = "0.1.0"

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 确保其他模块的日志也设置为 INFO 级别
logging.getLogger('tts_controller').setLevel(logging.INFO)
