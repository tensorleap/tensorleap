import os

max_sequence_length = 384  # The maximum length of a feature (question and context)
max_answer_length = 20
LABELS = ["start_idx", "end_idx"]
PAD_TOKEN = ''

# Preprocess Function
home_dir = os.getenv("HOME")
persistent_dir = os.path.join(home_dir, "Tensorleap", 'ALBERTqa')
TRAIN_SIZE = 1000
VAL_SIZE = 1000
# test_size = 500  # 1000 100
CHANGE_INDEX_FLAG = True

input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']