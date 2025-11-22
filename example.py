import random

inputs = []
outputs = []
data_samples = 10_000
for _ in range(data_samples):
	# Random number of packets sent for each data sample
	inputs.append(random.randint(1, 20))
	# Random number of devices that the user owns
	outputs.append(random.randint(1, 10))

def generate_sample():
    region = random.choice(["US", "EU", "ASIA"])
    session_id = random.randint(100000, 999999)
    packets = random.randint(1, 50)
    avg_size = random.uniform(100, 1500)
    avg_time = random.uniform(0.2, 5)

	input_data = InputData(region, session_id, packets, avg_size, avg_time, 0.1, 7.2, 5, 3)


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
        self.packets += random.randint(-3, 10)
        self.avg_time += random.uniform(-0.5, 2)

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
