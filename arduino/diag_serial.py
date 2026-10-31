import serial, time, sys

PORT = 'COM5'
BAUD = 115200

def log(tag, data):
    if data:
        print(f"[{tag}] {data.decode('utf-8', errors='replace')}", end='', flush=True)

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.2)
except Exception as e:
    print("OPEN FAILED:", e)
    sys.exit(1)

# Force a clean reset: drop DTR then raise it.
ser.dtr = False
time.sleep(0.2)
ser.dtr = True
ser.reset_input_buffer()

t0 = time.time()

def read_for(seconds, tag):
    end = time.time() + seconds
    while time.time() < end:
        data = ser.read(4096)
        if data:
            log(tag, data)

print("=== phase 1: startup + first auto episode (15 s) ===")
read_for(15, "boot")

print("\n=== phase 2: manual open test 'o 2500' ===")
ser.write(b"o 2500\r\n")
read_for(5, "open")

print("\n=== phase 3: manual close test 'c 2500' ===")
ser.write(b"c 2500\r\n")
read_for(5, "close")

print("\n=== phase 4: help ===")
ser.write(b"?\r\n")
read_for(2, "help")

ser.close()
print("\nDONE")
