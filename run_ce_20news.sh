gpu=$1
echo "ITERATION-1"
echo "Running GPT2 CE iteration 1"
python3 gpt2_ce.py 1 ${gpu} 1 20news
echo "Generating Computer Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} computer 2000 20news
echo "Training Computer BERT.."
python3 bert_train.py 1 ${gpu} 1 computer 1 20news
echo "Generating Recreation Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} recreation 2000 20news
echo "Training Recreation BERT.."
python3 bert_train.py 1 ${gpu} 1 recreation 1 20news
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 2000 20news
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 1 science 1 20news
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 2000 20news
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 1 politics 1 20news
echo "Generating Religion Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} religion 2000 20news
echo "Training Religion BERT.."
python3 bert_train.py 1 ${gpu} 1 religion 1 20news
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE iteration 2"
python3 gpt2_ce.py 1 ${gpu} 2 20news
echo "Generating Computer Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} computer 2000 20news
echo "Training Computer BERT.."
python3 bert_train.py 1 ${gpu} 2 computer 1 20news
echo "Generating Recreation Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} recreation 2000 20news
echo "Training Recreation BERT.."
python3 bert_train.py 1 ${gpu} 2 recreation 1 20news
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 2000 20news
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 2 science 1 20news
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 2000 20news
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 2 politics 1 20news
echo "Generating Religion Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} religion 2000 20news
echo "Training Religion BERT.."
python3 bert_train.py 1 ${gpu} 2 religion 1 20news