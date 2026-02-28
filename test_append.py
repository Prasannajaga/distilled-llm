import numpy as np
import os
import struct

with open('test_append.bin', 'wb') as f:
    f.write(b"TBIN")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<I", 512))
    f.write(struct.pack("<I", 2))
    for i in range(10):
        buffer = [1, 2, 3, 4]
        np.array(buffer, dtype=np.uint16).tofile(f)

print(os.path.getsize('test_append.bin'))
