gpu=$1
echo "ITERATION-1"
echo "Running GPT2 CE computer iteration 1"
python3 gpt2_ce_no_inter.py 1 ${gpu} 1 20news ce computer
echo "Generating Computer Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} computer 2000 20news ce
echo "Training Computer BERT.."
python3 bert_train.py 1 ${gpu} 1 computer 1 20news ce
echo "Running GPT2 CE recreation iteration 1"
python3 gpt2_ce_no_inter.py 1 ${gpu} 1 20news ce recreation
echo "Generating Recreation Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} recreation 2000 20news ce
echo "Training Recreation BERT.."
python3 bert_train.py 1 ${gpu} 1 recreation 1 20news ce
echo "Running GPT2 CE science iteration 1"
python3 gpt2_ce_no_inter.py 1 ${gpu} 1 20news ce science
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 2000 20news ce
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 1 science 1 20news ce
echo "Running GPT2 CE politics iteration 1"
python3 gpt2_ce_no_inter.py 1 ${gpu} 1 20news ce politics
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 2000 20news ce
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 1 politics 1 20news ce
echo "Running GPT2 CE religion iteration 1"
python3 gpt2_ce_no_inter.py 1 ${gpu} 1 20news ce religion
echo "Generating Religion Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} religion 2000 20news ce
echo "Training Religion BERT.."
python3 bert_train.py 1 ${gpu} 1 religion 1 20news ce
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE computer iteration 2"
python3 gpt2_ce_no_inter.py 1 ${gpu} 2 20news ce computer
echo "Generating Computer Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} computer 2000 20news ce
echo "Training Computer BERT.."
python3 bert_train.py 1 ${gpu} 2 computer 1 20news ce
echo "Running GPT2 CE recreation iteration 2"
python3 gpt2_ce_no_inter.py 1 ${gpu} 2 20news ce recreation
echo "Generating Recreation Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} recreation 2000 20news ce
echo "Training Recreation BERT.."
python3 bert_train.py 1 ${gpu} 2 recreation 1 20news ce
echo "Running GPT2 CE science iteration 2"
python3 gpt2_ce_no_inter.py 1 ${gpu} 2 20news ce science
echo "Generating Science Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} science 2000 20news ce
echo "Training Science BERT.."
python3 bert_train.py 1 ${gpu} 2 science 1 20news ce
echo "Running GPT2 CE politics iteration 2"
python3 gpt2_ce_no_inter.py 1 ${gpu} 2 20news ce politics
echo "Generating Politics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} politics 2000 20news ce
echo "Training Politics BERT.."
python3 bert_train.py 1 ${gpu} 2 politics 1 20news ce
echo "Running GPT2 CE religion iteration 2"
python3 gpt2_ce_no_inter.py 1 ${gpu} 2 20news ce religion
echo "Generating Religion Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} religion 2000 20news ce
echo "Training Religion BERT.."
python3 bert_train.py 1 ${gpu} 2 religion 1 20news ce
