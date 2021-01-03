gpu=$1
dataset=$2
echo "ITERATION-1"
echo "Running GPT2 CE-only iteration 1"
python3 gpt_2_fine_ceonly_ablation.py 1 ${gpu} 1 ${dataset}
echo "Running GPT2 Fine-tune iteration 1"
python3 gpt_2_finetune_politics_prestart_prestart.py 1 ${gpu} ${gpu} ${dataset} 1
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} politics 250
python3 bert_train.py 1 ${gpu} 1 politics 1
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-only iteration 2"
python3 gpt_2_fine_ceonly_ablation.py 1 ${gpu} 2 ${dataset}
echo "Running GPT2 Fine-tune iteration 2"
python3 gpt_2_finetune_politics_prestart.py 1 ${gpu} ${gpu} ${dataset} 2
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} politics 250
python3 bert_train.py 1 ${gpu} 2 politics 1
echo "***********************************************************************"
echo "ITERATION-3"
echo "Running GPT2 CE-only iteration 3"
python3 gpt_2_fine_ceonly_ablation.py 1 ${gpu} 3 ${dataset}
echo "Running GPT2 Fine-tune iteration 3"
python3 gpt_2_finetune_politics_prestart.py 1 ${gpu} ${gpu} ${dataset} 3
echo "Generating Data.."
python3 generate_data.py 1 ${gpu} politics 250
python3 bert_train.py 1 ${gpu} 3 politics 1
