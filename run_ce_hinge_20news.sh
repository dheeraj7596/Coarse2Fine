gpu=$1
echo "ITERATION-1"
echo "Running GPT2 CE-Hinge iteration 1"
python3 gpt2_ce_hinge.py 1 ${gpu} 1 20news
echo "Generating Computer Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} computer 500 20news
echo "Training Computer BERT.."
python3 bert_train.py 1 ${gpu} 1 computer 1 20news
echo "Generating Recreation Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} recreation 250 20news
echo "Training Recreation BERT.."
python3 bert_train.py 1 ${gpu} 1 recreation 1 20news
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 90 20news
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 1 science 1 20news
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 500 20news
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 1 politics 1 20news
echo "Generating Religion Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} religion 2000 20news
echo "Training Religion BERT.."
python3 bert_train.py 1 ${gpu} 1 religion 1 20news
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-Hinge iteration 2"
python3 gpt2_ce_hinge.py 1 ${gpu} 2 20news
echo "Generating Arts Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} arts 500 20news
echo "Training Arts BERT.."
python3 bert_train.py 1 ${gpu} 2 arts 1 20news
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 250 20news
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 2 politics 1 20news
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 90 20news
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 2 science 1 20news
echo "Generating Business Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} business 500 20news
echo "Training Business BERT.."
python3 bert_train.py 1 ${gpu} 2 business 1 20news
echo "Generating Sports Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} sports 2000 20news
echo "Training Sports BERT.."
python3 bert_train.py 1 ${gpu} 2 sports 1 20news
echo "***********************************************************************"
echo "ITERATION-3"
echo "Running GPT2 CE-Hinge iteration 3"
python3 gpt2_ce_hinge.py 1 ${gpu} 3 20news
echo "Generating Arts Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} arts 500 20news
echo "Training Arts BERT.."
python3 bert_train.py 1 ${gpu} 3 arts 1 20news
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 250 20news
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 3 politics 1 20news
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 90 20news
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 3 science 1 20news
echo "Generating Business Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} business 500 20news
echo "Training Business BERT.."
python3 bert_train.py 1 ${gpu} 3 business 1 20news
echo "Generating Sports Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} sports 2000 20news
echo "Training Sports BERT.."
python3 bert_train.py 1 ${gpu} 3 sports 1 20news
