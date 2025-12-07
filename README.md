# Medical_Device_Noise_Injection
Code / Other Resources for CSS 538 Research Project

## Usage

How to use the data generation python script

Basic invocation (prints 5 lines of preview of generated samples):

```
python .\data_gen.py --preview 5
```

Enable noise injection (defaults to gaussian noise) at ~10% noise:

```
python .\data_gen.py --noise --noise-level 0.1 --preview 5
```

Use uniform noise instead:

```
python .\data_gen.py --noise --noise-type uniform --noise-level 0.05 --preview 5
```

You can specify a seed for reproducible results:

```
python .\data_gen.py --seed 42 --preview 3
```

Command-line flags
- `--samples`: Number of samples to generate (default 10000)
- `--noise`: Enable noise injection (off by default)
- `--noise-type`: `gaussian` or `uniform` (default `gaussian`)
- `--noise-level`: Relative noise magnitude as a fraction (default 0.1)
- `--seed`: Integer seed for deterministic RNG
- `--preview`: Number of sample pairs to print and exit (default 5)
- `--profile <name>` to generate samples from a named profile.
- `--list-profiles` to list available profiles.

Demographics and encodings
- `region` is numeric (1..5) where 1=NA, 2=EU, 3=AS, 4=SA, 5=AF.
- `gender` is now encoded as an integer: 0 = M, 1 = F, 2 = O.
- `condition` is now encoded as an integer: 0 = healthy, 1 = arrhythmia, 2 = hypertension.

Output data of the samples is stored in the `outputs` list, but we can rewire this however works best for input into the NN