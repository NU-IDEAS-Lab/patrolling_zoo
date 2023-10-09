#!/bin/bash

# Run all scripts in background, and pipe their output to files.
python ./aamas2024/train_1_attritionYesComms01SkipAsync.py > ./aamas2024/log/train_1_attritionYesComms01SkipAsync.log 2>&1 &
python ./aamas2024/train_2_attritionNoComms01SkipAsync.py > ./aamas2024/log/train_2_attritionNoComms01SkipAsync.log 2>&1 &
python ./aamas2024/train_3_attritionNoCommsNoSkipAsync.py > ./aamas2024/log/train_3_attritionNoCommsNoSkipAsync.log 2>&1 &
python ./aamas2024/train_4_attritionNoCommsNoSkipSync.py > ./aamas2024/log/train_4_attritionNoCommsNoSkipSync.log 2>&1 &
python ./aamas2024/train_5_attritionNoCommsNoSkipNo.py > ./aamas2024/log/train_5_attritionNoCommsNoSkipNo.log 2>&1 &
python ./aamas2024/train_6_attritionYesCommsNoSkipAsync.py > ./aamas2024/log/train_6_attritionYesCommsNoSkipAsync.log 2>&1 &
python ./aamas2024/train_7_attritionNoCommsNoSkipAsyncBadAlphaBeta.py > ./aamas2024/log/train_7_attritionNoCommsNoSkipAsyncBadAlphaBeta.log 2>&1 &
python ./aamas2024/train_8_attritionNoCommsNoSkipAsyncBadAlphaBeta2.py > ./aamas2024/log/train_8_attritionNoCommsNoSkipAsyncBadAlphaBeta2.log 2>&1 &