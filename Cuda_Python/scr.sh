#!/bin/bash

for i in $(seq 1 5)
do
	LD_BIND_NOW=1 ./testSum
done

for i in $(seq 1 5)
do
	LD_BIND_NOW=1 python3 ./pySum.py
done
