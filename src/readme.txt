1. SCSGan.py: build, train, and test the SCS-Gan model.

2. read_dataset.py: to read and generate data used in training and testing stages.
    - tfr.py: called by read_dataset for generating datasets.
    - utils.py: basice functions used in the whole process.
    
3. tokenizer_sp.py: to generate and use SentencePiece tokenizer [1].

4. ./bert/: to fine-tune CodeBERT and use it for adversarial training purpose.



[1] Kudo, Taku, and John Richardson. "Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing." arXiv preprint arXiv:1808.06226 (2018).