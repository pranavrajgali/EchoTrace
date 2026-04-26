import ctypes
import os

dll_path = r"C:\Users\Admin\Documents\EchoTrace_V4\EchoTraceV2\EchoTrace\.venv\Lib\site-packages\numpy\_core\_multiarray_umath.cp313-win_amd64.pyd"
print(f"Checking {dll_path}...")

if os.path.exists(dll_path):
    try:
        handle = ctypes.WinDLL(dll_path)
        print("Successfully loaded numpy pyd!")
    except Exception as e:
        print(f"Failed to load numpy pyd: {e}")
else:
    print("File does not exist!")
