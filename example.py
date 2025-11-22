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
	def __init__(self, region, session_id, packets, avg_size, avg_time, min_time, max_time, upload_batch, upload_single):
        self.region = region
        self.session_id = session_id
        self.packets = packets
        self.avg_size = avg_size
        self.avg_time = avg_time
        self.min_time = min_time
        self.max_time = max_time
        self.upload_batch = upload_batch
        self.upload_single = upload_single
	
	def inject_noise(self):
		self.packets += random.randint(5, 30)

class Packet:
	def __init__(self, src, dest, size, data = None):
		self.src = src
		self.dest = dest
		self.size = size
		self.data = data

class OutputData:
	def __init__(self, age, gender, condition, device_count):
        self.age = age
        self.gender = gender
        self.condition = condition
        self.device_count = device_count
