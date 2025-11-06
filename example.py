import random

inputs = []
outputs = []
data_samples = 10_000
for _ in range(data_samples):
	# Random number of packets sent for each data sample
	inputs.append(random.randint(1, 20))
	# Random number of devices that the user owns
	outputs.append(random.randint(1, 10))

class InputData:
	def __init__(self, packets):
		self.packets = packets
	
	def inject_noise(self):
		self.packets += random.randint(5, 30)

class Packet:
	def __init__(self, src, dest, size, data = None):
		self.src = src
		self.dest = dest
		self.size = size
		self.data = data

class OutputData:
	def __init__(self, devices, condition_type):
		self.devices = devices
		self.condition_type = condition_type