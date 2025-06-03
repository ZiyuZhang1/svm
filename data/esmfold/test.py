# First verify where PIL is coming from
import sys
import os
print("Python executable:", sys.executable)
print("Python path:", sys.path)

# Try importing from the environment's PIL
import PIL
print("PIL path:", PIL.__file__)

# Now try the problematic import
from PIL import _imaging
print("_imaging imported successfully")