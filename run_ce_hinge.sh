gpu=$1
echo "ITERATION-1"
echo "Running GPT2 CE-Hinge iteration 1"
python3 gpt2_ce_hinge.py 1 ${gpu} 1 nyt ce_hinge
echo "Generating Arts Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} arts 500 nyt ce_hinge
echo "Training Arts BERT.."
python3 bert_train.py 1 ${gpu} 1 arts 1 nyt ce_hinge
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 250 nyt ce_hinge
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 1 politics 1 nyt ce_hinge
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 90 nyt ce_hinge
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 1 science 1 nyt ce_hinge
echo "Generating Business Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} business 500 nyt ce_hinge
echo "Training Business BERT.."
python3 bert_train.py 1 ${gpu} 1 business 1 nyt ce_hinge
echo "Generating Sports Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} sports 2000 nyt ce_hinge
echo "Training Sports BERT.."
python3 bert_train.py 1 ${gpu} 1 sports 1 nyt ce_hinge
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE-Hinge iteration 2"
python3 gpt2_ce_hinge.py 1 ${gpu} 2 nyt ce_hinge
echo "Generating Arts Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} arts 500 nyt ce_hinge
echo "Training Arts BERT.."
python3 bert_train.py 1 ${gpu} 2 arts 1 nyt ce_hinge
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 250 nyt ce_hinge
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 2 politics 1 nyt ce_hinge
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 90 nyt ce_hinge
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 2 science 1 nyt ce_hinge
echo "Generating Business Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} business 500 nyt ce_hinge
echo "Training Business BERT.."
python3 bert_train.py 1 ${gpu} 2 business 1 nyt ce_hinge
echo "Generating Sports Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} sports 2000 nyt ce_hinge
echo "Training Sports BERT.."
python3 bert_train.py 1 ${gpu} 2 sports 1 nyt ce_hinge
echo "***********************************************************************"
