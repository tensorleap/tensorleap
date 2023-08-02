import os
from code_loader import LeapLoader

def check_intagration():
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = 'tensorleap.py'
    datascript = LeapLoader(code_path=dir_path, code_entry_name=script_path)
    res = datascript.check_dataset()
    if res.is_valid:
        print('Integration script is valid')
    else:
        print(f"Integration failed with error: {res.general_error}")

if __name__ == '__main__':
    check_intagration()