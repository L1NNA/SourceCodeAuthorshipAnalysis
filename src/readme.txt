1. ./SCSGan/SCSGan.py: build, train, and test the SCS-Gan model.

2. ./SCSGan/read_dataset.py: to read and generate data used in training and testing stages.
    - ./SCSGan/tfr.py: called by read_dataset for generating datasets.
    - ./SCSGan/utils.py: basice functions used in the whole process.
    
3. ./SCSGan/tokenizer_sp.py: to generate and use SentencePiece tokenizer [1].

4. ./CodeBERT/: to fine-tune CodeBERT and use it for adversarial training purpose. 
   For details please see the readme file in ./CodeBERT.



[1] Kudo, Taku, and John Richardson. "Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing." arXiv preprint arXiv:1808.06226 (2018).