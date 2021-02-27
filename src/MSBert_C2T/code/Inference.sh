data_dir='/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T'
output_dir='/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T'

batch_size=10
test_file=$data_dir/gcj2017_py.jsonl
source_length=256
target_length=128
beam_size=10

# python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size