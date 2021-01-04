gpu=$1
dataset=$2
num=$3
echo "ITERATION-1"
echo "Running GPT2 CE-only iteration 1"
python3 gpt2_fine_ceonly_ablation.py 1 ${gpu} 1 ${dataset} fine_ceonly
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num}
echo "Training BERT.."
python3 bert_train_ceonly.py 1 ${gpu} 1 ${dataset} 1
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-only iteration 2"
python3 gpt2_fine_ceonly_ablation.py 1 ${gpu} 2 ${dataset} fine_ceonly
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num}
echo "Training BERT.."
python3 bert_train_ceonly.py 1 ${gpu} 2 ${dataset} 1
echo "***********************************************************************"
echo "ITERATION-3"
echo "Running GPT2 CE-only iteration 3"
python3 gpt2_fine_ceonly_ablation.py 1 ${gpu} 3 ${dataset} fine_ceonly
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} ${dataset} ${num}
echo "Training BERT.."
python3 bert_train_ceonly.py 1 ${gpu} 3 ${dataset} 1
