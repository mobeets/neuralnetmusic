
## Requirements

* Theano
* numpy
* keras

## Usage

Generate a four-bar sample:

```bash
$ python DBN.py sample
```

Harmonize with an existing four-bar sample:

```bash
$ python DBN.py harmonize path/to/song.midi
```

## Playing MIDI

```bash
$ fluidsynth -i input/soundfontfile.sf2 output/test2.midi
```
