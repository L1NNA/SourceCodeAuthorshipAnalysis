#!/usr/bin/env bash

TEST_DIR=/home/jovyan/Source_Code_Veri/data/GCJ/C2V/files/2016/py
OUT_DATA_BASE=/home/jovyan/Source_Code_Veri/data/GCJ/C2V
DATASET_NAME=gcj2016_py

MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
NUM_THREADS=64
PYTHON=python3
###########################################################

TEST_DATA_FILE=${DATASET_NAME}.test.raw.txt

# java extractor
# EXTRACTOR_JAR=JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar
# python extractor


mkdir -p ${OUT_DATA_BASE}/${DATASET_NAME}
mkdir -p data/${DATASET_NAME}

echo "Extracting paths from test set..."


OtherExtractor/astminer/cli.sh code2vec --lang py --project ${TEST_DIR} --output data/${DATASET_NAME}/${TEST_DATA_FILE} --maxL MAX_CONTEXTS --maxW WORD_VOCAB_SIZE --maxContexts MAX_CONTEXTS --maxTokens TARGET_VOCAB_SIZE --maxPaths PATH_VOCAB_SIZE  --split-tokens 



cat data/${DATASET_NAME}/${TEST_DATA_FILE} | wc -l
echo "Finished extracting paths from test set"

TARGET_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.path.c2v

echo "Creating histograms from the training data"
cat data/${DATASET_NAME}/${TEST_DATA_FILE} | cut -d' ' -f1 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat data/${DATASET_NAME}/${TEST_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${ORIGIN_HISTOGRAM_FILE}
cat data/${DATASET_NAME}/${TEST_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}

${PYTHON} preprocess.py --data_file data/${DATASET_NAME}/${TEST_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --word_vocab_size ${WORD_VOCAB_SIZE} --path_vocab_size ${PATH_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --word_histogram ${ORIGIN_HISTOGRAM_FILE} \
  --path_histogram ${PATH_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name ${OUT_DATA_BASE}/${DATASET_NAME}/${DATASET_NAME}
    
# If all went well, the raw data files can be deleted, because preprocess.py creates new files 
# with truncated and padded number of paths for each example.
# rm ${TEST_DATA_FILE} ${TARGET_HISTOGRAM_FILE} ${ORIGIN_HISTOGRAM_FILE} \
#   ${PATH_HISTOGRAM_FILE}

