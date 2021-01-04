gpu=$1
dataset=$2
num=$3
echo "ITERATION-1"
echo "Running GPT2 KL-div-CE iteration 1"
python3 gpt2_fine_finetune.py 1 ${gpu} ${gpu} ${dataset} 1
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num} fine
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 1 ${dataset} 1
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 KL-div-CE iteration 2"
python3 gpt2_fine_finetune.py 1 ${gpu} ${gpu} ${dataset} 2
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num} fine
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 2 ${dataset} 1
echo "***********************************************************************"
echo "ITERATION-3"
echo "Running GPT2 KL-div-CE iteration 3"
python3 gpt2_fine_finetune.py 1 ${gpu} ${gpu} ${dataset} 3
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num} fine
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 3 ${dataset} 1
