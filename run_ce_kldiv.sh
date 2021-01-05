gpu=$1
plabel=$2
num=$3
dataset=$4
echo "ITERATION-1"
echo "Running GPT2 CE-KL-div iteration 1"
python3 gpt2_finetune_ce_kl_div.py.py 1 ${gpu} 1 ${plabel} fine ${dataset}
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${plabel} ${num} fine ${dataset}
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 1 ${plabel} 1 ${dataset}
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-KL-div iteration 2"
python3 gpt2_finetune_ce_kl_div.py.py 1 ${gpu} 2 ${plabel} fine ${dataset}
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${plabel} ${num} fine ${dataset}
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 2 ${plabel} 1 ${dataset}
echo "***********************************************************************"
echo "ITERATION-3"
echo "Running GPT2 CE-KL-div iteration 3"
python3 gpt2_finetune_ce_kl_div.py.py 1 ${gpu} 3 ${plabel} fine ${dataset}
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${plabel} ${num} fine ${dataset}
echo "Training BERT.."
python3 bert_train.py 1 ${gpu} 3 ${plabel} 1 ${dataset}
