#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
#   each of {TRAIN_DIR, VAL_DIR and TEST_DIR} should have sub-dirs,
#   and data will be extracted from .java files found in those sub-dirs).
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_CONTEXTS is the number of contexts to keep for each 
#   method (by default 200).                              
# WORD_VOCAB_SIZE, PATH_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of words, paths and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
#   The default values are reasonable for a Tesla K80 GPU 
#   and newer (12 GB of board memory).
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
# TRAIN_DIR=/home/jovyan/Source_Code_Veri/data/GCJ/files/2017/java
# VAL_DIR=/home/jovyan/Source_Code_Veri/data/GCJ/files/2017/java

TEST_DIR=/home/jovyan/Source_Code_Veri/data/GCJ/C2V/files/2017/tmp_java
OUT_DATA_BASE=/home/jovyan/Source_Code_Veri/data/GCJ/C2V
DATASET_NAME=GCJ_2017_java
MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
NUM_THREADS=64
PYTHON=python3
###########################################################

# TRAIN_DATA_FILE=${DATASET_NAME}.train.raw.txt
# VAL_DATA_FILE=${DATASET_NAME}.val.raw.txt
TEST_DATA_FILE=${DATASET_NAME}.test.raw.txt

# java extractor
# EXTRACTOR_JAR=JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar
# python extractor
EXTRACTOR_JAR=JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar


mkdir -p ${OUT_DATA_BASE}/${DATASET_NAME}
mkdir -p data/${DATASET_NAME}

# echo "Extracting paths from validation set..."
# ${PYTHON} JavaExtractor/extract.py --dir ${VAL_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${VAL_DATA_FILE}
# echo "Finished extracting paths from validation set"
echo "Extracting paths from test set..."
${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > data/${DATASET_NAME}/${TEST_DATA_FILE}

cat data/${DATASET_NAME}/${TEST_DATA_FILE} | wc -l
echo "Finished extracting paths from test set"
# echo "Extracting paths from training set..."
# ${PYTHON} JavaExtractor/extract.py --dir ${TRAIN_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} | shuf > ${TRAIN_DATA_FILE}
# echo "Finished extracting paths from training set"

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

