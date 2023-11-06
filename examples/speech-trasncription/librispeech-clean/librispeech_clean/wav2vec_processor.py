from transformers import Wav2Vec2Processor
from librispeech_clean.configuration import config


class ProcessorSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessorSingleton, cls).__new__(cls)
            cls._instance.processor = Wav2Vec2Processor.from_pretrained(config.get_parameter('model_hf_id'))
        return cls._instance

    def get_processor(self):
        return self.processor
