import sounddevice as sd
print(sd.query_devices())
print("\nYour default input device index is:", sd.default.device[0])