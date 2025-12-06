import random
import argparse
import json
from typing import Tuple

inputs = []
outputs = []

# Numeric mappings for demographics for use with TNN to avoid strings
# gender: 0 = M, 1 = F, 2 = O
# condition: 0 = healthy, 1 = arrhythmia, 2 = hypertension
# region: 1 = NA, 2 = EU, 3 = AS, 4 = SA, 5 = AF

def generate_sample(rng: random.Random) -> Tuple['InputData', 'OutputData']:
    """Generate a single synthetic input/output pair using provided RNG.

    Args:
        rng: a random.Random instance for reproducible output.

    Returns:
        (InputData, OutputData)
    """
    region = rng.randint(1, 5)
    session_id = rng.randint(100000, 999999)
    packets = rng.randint(1, 50)
    avg_size = rng.uniform(100, 1500)
    avg_time = rng.uniform(0.2, 5)

    input_data = InputData(region, session_id, packets, avg_size, avg_time, 0.1, 7.2, 5, 3)

    # Demographic data
    age = rng.randint(18, 90)
    gender = rng.randint(0, 2)
    condition = rng.randint(0, 2)
    device_count = rng.randint(1, 10)

    output_data = OutputData(age, gender, condition, device_count)
    return input_data, output_data


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

    def inject_noise(self, noise_type: str = 'gaussian', level: float = 0.1, rng: random.Random = None):
        """Inject noise into numeric fields.

        noise_type: 'gaussian' or 'uniform'
        level: relative noise magnitude (fractional). For gaussian, treated as relative stddev; for uniform, as max relative amplitude.
        rng: random.Random instance for deterministic noise.
        """
        if rng is None:
            rng = random

        if noise_type == 'gaussian':
            delta = int(round(rng.gauss(0, max(1.0, level * self.packets))))
        else:  # uniform
            delta = int(round(rng.uniform(-level * self.packets, level * self.packets)))
        self.packets = max(1, self.packets + delta)

        # avg_size (bytes)
        if noise_type == 'gaussian':
            self.avg_size = max(1.0, self.avg_size + rng.gauss(0, level * self.avg_size))
        else:
            self.avg_size = max(1.0, self.avg_size + rng.uniform(-level * self.avg_size, level * self.avg_size))

        # avg_time (s)
        if noise_type == 'gaussian':
            self.avg_time = max(0.0, self.avg_time + rng.gauss(0, level * self.avg_time))
        else:
            self.avg_time = max(0.0, self.avg_time + rng.uniform(-level * self.avg_time, level * self.avg_time))

    def to_dict(self):
        return {
            'region': self.region,
            'session_id': self.session_id,
            'packets': self.packets,
            'avg_size': self.avg_size,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'upload_batch': self.upload_batch,
            'upload_single': self.upload_single,
        }

    def __repr__(self):
        return f"InputData(region={self.region}, session_id={self.session_id}, packets={self.packets}, avg_size={self.avg_size:.2f}, avg_time={self.avg_time:.3f})"

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

    def to_dict(self):
        return {
            'age': self.age,
            'gender': self.gender,
            'condition': self.condition,
            'device_count': self.device_count,
        }

    def __repr__(self):
        return f"OutputData(age={self.age}, gender={self.gender}, condition={self.condition}, device_count={self.device_count})"


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic samples. Optionally inject noise.')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--noise', action='store_true', help='Enable noise injection')
    parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian', help='Noise distribution to use')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Relative noise level (fraction). 0.1 means ~10%%')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible samples')
    parser.add_argument('--preview', type=int, default=5, help='Print this many sample pairs to stdout and exit')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    samples = []
    for _ in range(args.samples):
        inp, out = generate_sample(rng)
        if args.noise:
            inp.inject_noise(noise_type=args.noise_type, level=args.noise_level, rng=rng)
        samples.append((inp, out))

    inputs.clear()
    outputs.clear()
    for inp, out in samples:
        inputs.append(inp)
        outputs.append(out)

    # Print a small preview (JSON) for debugging
    for inp, out in samples[:args.preview]:
        print(json.dumps({'input': inp.to_dict(), 'output': out.to_dict()}))


if __name__ == '__main__':
    main()
