#! /bin/bash

echo "@@@@@@@@@@@@@@@@@"
echo "@ Prepare setup @"
echo "@@@@@@@@@@@@@@@@@\n"

python3 prepare.py

echo "\n@@@@@@@@@@@@@@@@@"
echo "@ Estimate SNR  @"
echo "@@@@@@@@@@@@@@@@@\n"

python3 compute_snr.py

echo "\n@@@@@@@@@@@@@@@@@"
echo "@ Model Shares  @"
echo "@@@@@@@@@@@@@@@@@\n"

python3 modeling.py

echo "\n@@@@@@@@@@@@@@@@@"
echo "@ Recover Key   @"
echo "@@@@@@@@@@@@@@@@@\n"

python3 attack.py
