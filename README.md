# pyvisgen [![Actions Status](https://github.com/radionets-project/pyvisgen/workflows/CI/badge.svg)](https://github.com/radionets-project/pyvisgen/actions)

Python implementation of the VISGEN tool developed at [Haystack Observatory](https://www.haystack.mit.edu/astronomy/). It uses the Radio Interferometer Measurement Equation (RIME) to simulate the measurement process of a radio interferometer. A gridder is also implemented to process the resulting visibilities and convert them to images suitable as input for the neural networks developed in the [radionets repository](https://github.com/radionets-project/radionets).

## Usage
There are 3 possible modes at the moment:  `simulate` (default), `slurm`, and `gridding`. `simulate` and `slurm` both utilize the RIME formalism for creating visibilities data. With the option `gridding`, these visibilities get gridded and prepared as input images for training a neural network from the radionets framework.
```
$ pyvisgen_create_dataset --mode=simulate some_file.toml
```

Visualization of the E matrix
![visualize_E](https://github.com/radionets-project/pyvisgen/assets/23259659/194a321b-77cd-423b-9d01-c18c0741d6c5)

Visualization of the K matrix
![visualize_K](https://github.com/radionets-project/pyvisgen/assets/23259659/501f487a-498b-4143-b54a-eb0e2f28e417)
