import ctypes
import os
import pathlib

dll_path = r"C:\Users\Admin\Documents\EchoTrace_V4\EchoTraceV2\EchoTrace\.venv\Lib\site-packages\torch\lib\torch_python.dll"
print(f"Checking {dll_path}...")

if os.path.exists(dll_path):
    print("File exists.")
    try:
        # Try to load it
        handle = ctypes.WinDLL(dll_path)
        print("Successfully loaded DLL!")
    except Exception as e:
        print(f"Failed to load DLL: {e}")
else:
    print("File does not exist!")
