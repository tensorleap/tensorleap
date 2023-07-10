import os
from code_loader import LeapLoader

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = 'tensorleap.py'
    datascript = LeapLoader(code_path=dir_path, code_entry_name=script_path)
    res = datascript.check_dataset()
    assert res.is_valid
