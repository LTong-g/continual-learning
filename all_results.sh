#!/usr/bin/env bash

########### Three Types of Incremental Learning (2022, Nat Mach Intell) ###########

## MNIST
./compare_hyperParams.py --seed=1 --experiment=splitMNIST
./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST
./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --tau-per-budget


## Pre-training on CIFAR-10
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=1
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=2
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=3
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=4
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=5
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=6
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=7
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=8
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=9
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=10
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=11


## CIFAR-100
./compare_hyperParams.py --seed=1 --experiment=CIFAR100 --pre-convE --freeze-convE --seed-to-ltag --no-fromp
./compare.py --seed=2 --n-seeds=10 --experiment=CIFAR100 --pre-convE --freeze-convE --no-fromp --seed-to-ltag --eval-s=10000
./compare_replay.py --seed=2 --n-seeds=5 --experiment=CIFAR100 --pre-convE --freeze-convE --seed-to-ltag --no-fromp
