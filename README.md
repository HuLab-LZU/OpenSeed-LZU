# OpenSeed-LZU

Towards a open-source seed image analysis ecosystem.

## Getting started

1. install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. clone this repo
```bash
git clone https://github.com/HuLab-LZU/OpenSeed-LZU.git OpenSeed-LZU
cd OpenSeed-LZU
```
3. install dependencies
```bash
uv sync
```
4. activate virtual environment
```bash
source .venv/bin/activate
```
5. training
OpenSeed employs [Hydra](https://github.com/facebookresearch/hydra) to train models, get more
training commandline help from [Hydra Document](https://hydra.cc/docs/intro/).

Some scripts are also provided to simplify this, you can find them in [scripts](scripts) directory.

```bash
./scripts/run.sh mobilenet conv_small -ds rgb -d '[0]' -b 256 -dd <dataset path> --lr 0.0001
```
explore more options from the scripts or using `python src/train.py --help`
or ``python src/repeat.py --help``

## Cite

TODO: add citation info.
