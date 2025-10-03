# EMaGerZ paper

This repository is used for offline training and evaluation. It also serves as the central hub towards other repositories for the EMaGerZ platform.

## Setup

First, setup the globals in `globals.py`.

## Linked repositories

- The central library for the embedded and workstation codebases: [emager-py](https://github.com/SBIOML/emager-py)
- RTL module for [RHD2164](https://github.com/SBIOML/rhd2164-spi-fpga)
- HAL for RHD2000 chips: [librhd](https://github.com/SBIOML/rhd2000-driver)
- Sampling executable for PYNQ: [rhd-sampler](https://github.com/SBIOML/rhd-sampler)
- The embedded platform gesture recognition implementation: [emager-pynq](https://github.com/SBIOML/emager-pynq)
After, you can experiment with `stats.py` to visualize the models' performance.

## rhd sampler evaluation

1. run rhd-sampler: `sudo rhd_sampler` (make sure the redis keys are ok)
2. find PID: `PID=$(pgrep -f rhd_ | tail -n 1)`
3. get statistics: `sudo cat "/proc/${PID}/smaps" | grep -i pss | awk '{total += $2} END {print total " kB"}'`

## VirtualBox FINN build

Create the necessary folders:

```bash
mkdir ~/Xilinx
mkdir ~/Documents/git/emager-pynq-paper
```
Share the `emager-pynq-paper` folder but do not auto-mount it. Same thing for Xilinx installation, name it `xilinx` (as the folder cannot have the same name as the shared folder name). Then, in the guest: 

```bash
sudo mount -t vboxsf -o uid=1000,gid=1000 paper ~/Documents/git/emager-pynq-paper
sudo mount -t vboxsf -o uid=1000,gid=1000 xilinx ~/Xilinx/
```
