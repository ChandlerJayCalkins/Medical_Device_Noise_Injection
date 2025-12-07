import random
import argparse
import json
from typing import Tuple, Dict, Any

inputs = []
outputs = []

# Defining some profiles for different users, arbitrarily chosen numbers
PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    # fewer packets, moderate sizes, low latency
    'healthy_young': {
        'packets': {'dist': 'normal', 'mean': 8, 'std': 3},
        'avg_size': {'dist': 'normal', 'mean': 400.0, 'std': 100.0},
        'avg_time': {'dist': 'normal', 'mean': 0.5, 'std': 0.2},
    },
    # slightly more packets, larger sizes, slightly higher times
    'healthy_elderly': {
        'packets': {'dist': 'normal', 'mean': 12, 'std': 4},
        'avg_size': {'dist': 'normal', 'mean': 600.0, 'std': 150.0},
        'avg_time': {'dist': 'normal', 'mean': 0.8, 'std': 0.3},
    },
    # many packets (frequent readings), small sizes, low times
    'arrhythmia_active': {
        'packets': {'dist': 'normal', 'mean': 30, 'std': 8},
        'avg_size': {'dist': 'normal', 'mean': 250.0, 'std': 80.0},
        'avg_time': {'dist': 'normal', 'mean': 0.3, 'std': 0.15},
    },
    # moderate packets, larger sizes, higher times
    'hypertension': {
        'packets': {'dist': 'normal', 'mean': 18, 'std': 6},
        'avg_size': {'dist': 'normal', 'mean': 900.0, 'std': 200.0},
        'avg_time': {'dist': 'normal', 'mean': 1.2, 'std': 0.4},
    },
}

# Numeric mappings for demographics for use with TNN to avoid strings
# gender: 0 = M, 1 = F, 2 = O
# condition: 0 = healthy, 1 = arrhythmia, 2 = hypertension
# region: 1 = NA, 2 = EU, 3 = AS, 4 = SA, 5 = AF

def _sample_from_spec(spec: Dict[str, Any], rng: random.Random) -> float:
    dist = spec.get('dist', 'normal')
    if dist == 'normal':
        mean = float(spec.get('mean', 0.0))
        std = float(spec.get('std', 1.0))
        return rng.gauss(mean, std)
    else:  # uniform
        lo = float(spec.get('min', 0.0))
        hi = float(spec.get('max', 1.0))
        return rng.uniform(lo, hi)


def generate_sample(rng: random.Random, profile_name: str = 'healthy_young') -> Tuple['InputData', 'OutputData']:
    """Generate a single synthetic input/output pair using provided RNG and profile.

    Args:
        rng: a random.Random instance for reproducible output.
        profile_name: key in `PROFILES` that determines distributions for
            `packets`, `avg_size`, and `avg_time`.

    Returns:
        (InputData, OutputData)
    """
    profile = PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_name}")

    region = rng.randint(1, 5)
    session_id = rng.randint(100000, 999999)

    # Sample the features from profile distribution
    packets_f = _sample_from_spec(profile['packets'], rng)
    packets = max(1, int(round(packets_f)))

    avg_size = max(1.0, float(_sample_from_spec(profile['avg_size'], rng)))
    avg_time = max(0.0, float(_sample_from_spec(profile['avg_time'], rng)))
    # Derive demographic data from profile
    if 'healthy' in profile_name:
        age = rng.randint(18, 40) if 'young' in profile_name else rng.randint(40, 85)
        condition = 0
    elif 'arrhythmia' in profile_name:
        age = rng.randint(30, 80)
        condition = 1
    elif 'hypertension' in profile_name:
        age = rng.randint(40, 85)
        condition = 2
    else:
        age = rng.randint(18, 90)
        condition = rng.randint(0, 2)

    gender = rng.randint(0, 2) # we could tie gender to a profile if needed
    device_count = rng.randint(1, 10)

    # Build InputData (packet/session info) and OutputData (profile/demographics)
    input_data = InputData(region, session_id, packets, avg_size, avg_time, 0.1, 7.2, 5, 3)
    output_data = OutputData(age, gender, condition, device_count)
    return input_data, output_data


class InputData:
    def __init__(self, region, session_id, packets, avg_size, avg_time):
        self.region = region
        self.session_id = session_id
        self.packets = packets
        self.avg_size = avg_size
        self.avg_time = avg_time

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
            'avg_time': self.avg_time
        }
    
    def to_list(self):
        return [
            self.region,
            self.session_id,
            self.packets,
            self.avg_size,
            self.avg_time
		]

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
    
    def to_list(self):
        return [
            self.age,
            self.gender,
            self.condition,
            self.device_count
		]

    def __repr__(self):
        return f"OutputData(age={self.age}, gender={self.gender}, condition={self.condition}, device_count={self.device_count})"

parser = argparse.ArgumentParser(description='Generate synthetic samples. Optionally inject noise.')
parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
parser.add_argument('--noise', action='store_true', help='Enable noise injection')
parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian', help='Noise distribution to use')
parser.add_argument('--noise-level', type=float, default=0.1, help='Relative noise level (fraction). 0.1 means ~10%%')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible samples')
parser.add_argument('--preview', type=int, default=5, help='Print this many sample pairs to stdout and exit')
parser.add_argument('--profile', type=str, default='healthy_young', help='Profile name to use for generation')
parser.add_argument('--list-profiles', action='store_true', help='List available generation profiles and exit')
args = parser.parse_args()

def main():
    if args.list_profiles:
        print(json.dumps({'profiles': list(PROFILES.keys())}, indent=2))
        return

    if args.profile not in PROFILES:
        raise SystemExit(f"Unknown profile '{args.profile}'. Use --list-profiles to see valid names.")

    rng = random.Random(args.seed)

    samples = []
    for _ in range(args.samples):
        inp, out = generate_sample(rng, profile_name=args.profile)
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
