# AlienHunters_VAE_V2
Updated VAE Architecture V2

This project was made to create a proof of concept for synthetic satellite imagery generation. All the code here is open source for anyone to use or base a model on.

## Usage

### vae.py
This is the main file used for training the model; depends on `pipeline.py`.
Inside the file, any variables under the `CONFIGURATION VARIABLES` or the `CONSTANTS CONFIGURATION` sections may be editted to alter functionality as necessary.

Note that a generated manifest is required for model training. See manifest specification below.

Usage:
- `python vae.py [arg]`

Note: Only one argument can be passed at once

Command Line Arguments:
- `--load [int]		| Load checkpoint model and start at specified epoch`
- `-l [int]			| Alias for --load`
- `--arch			| Shown model architecture only, do not execute`
- `-a 				| Alias for --arch`


### generator.py
This is a standalone file and is simply run from the command line. Its only output is a single generated image.
To alter the output specifications you must manually edit the values under the `CONFIGURATION VALUES` sections, as well as under the main `for` loop.

## Project Configuration
**Note:** This model was built to run on Ubuntu and may not function correctly on Windows based systems.

### Directory Setup
In order to properly run this program you must create a folder named 'model' in the base directory

### Necessary Files
- Training Manifest
- Validation Manifest

See manifest specification below.

### Project Versions
This project was built using the following python packages and versions; newer versions may work but are not guaranteed.
- Python 3.7.6
- Numpy 1.19.5
- Tensorflow 2.4.1
- Keras 2.4.3
- Matplotlib 3.1.3

## Manifest Specifications
In order to run `vae.py` you must generate your own manifest to load files from. A manifest should consist of a space `" "` seperated list of absolute filepaths. The manifest should also contain a space at the end. All manifest loading functions can be found in `pipeline.py`.