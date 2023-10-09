#!/bin/bash

# Run all scripts in background, and pipe their output to files.
python ./train_1_attritionYesComms01SkipAsync.py > ./log/train_1_attritionYesComms01SkipAsync.log 2>&1 &
python ./train_2_attritionNoComms01SkipAsync.py > ./log/train_2_attritionNoComms01SkipAsync.log 2>&1 &
python ./train_3_attritionNoCommsNoSkipAsync.py > ./log/train_3_attritionNoCommsNoSkipAsync.log 2>&1 &
python ./train_4_attritionNoCommsNoSkipSync.py > ./log/train_4_attritionNoCommsNoSkipSync.log 2>&1 &
python ./train_5_attritionNoCommsNoSkipNo.py > ./log/train_5_attritionNoCommsNoSkipNo.log 2>&1 &
python ./train_6_attritionYesCommsNoSkipAsync.py > ./log/train_6_attritionYesCommsNoSkipAsync.log 2>&1 &


# RUN LATER!!
python ./train_7_attritionNoCommsNoSkipAsyncBadAlphaBeta.py > ./log/train_7_attritionNoCommsNoSkipAsyncBadAlphaBeta.log 2>&1 &
python ./train_8_attritionNoCommsNoSkipAsyncBadAlphaBeta2.py > ./log/train_8_attritionNoCommsNoSkipAsyncBadAlphaBeta2.log 2>&1 &