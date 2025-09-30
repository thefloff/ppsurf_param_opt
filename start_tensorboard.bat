@echo off
REM Activates the .venv environment
CALL .\.venv\Scripts\activate.bat

REM Starts TensorBoard with the specified log directory
tensorboard --logdir=models