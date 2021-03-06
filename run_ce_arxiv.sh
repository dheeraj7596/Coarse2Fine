gpu=$1
echo "ITERATION-1"
echo "Running GPT2 CE iteration 1"
python3 gpt2_ce.py 1 ${gpu} 1 arxiv ce
echo "Generating CS Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} cs 6000 arxiv ce
echo "Training CS BERT.."
python3 bert_train.py 1 ${gpu} 1 cs 1 arxiv ce
echo "Generating Physics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} physics 7000 arxiv ce
echo "Training Physics BERT.."
python3 bert_train.py 1 ${gpu} 1 physics 1 arxiv ce
echo "Generating Math Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} math 12000 arxiv ce
echo "Training Math BERT.."
python3 bert_train.py 1 ${gpu} 1 math 1 arxiv ce
echo "***********************************************************************"
echo "ITERATION-2"
echo "Running GPT2 CE iteration 2"
python3 gpt2_ce.py 1 ${gpu} 2 arxiv ce
echo "Generating CS Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} cs 6000 arxiv ce
echo "Training CS BERT.."
python3 bert_train.py 1 ${gpu} 2 cs 1 arxiv ce
echo "Generating Physics Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} physics 7000 arxiv ce
echo "Training Physics BERT.."
python3 bert_train.py 1 ${gpu} 2 physics 1 arxiv ce
echo "Generating Math Data.."
python3 generate_data_ce_hinge.py 1 ${gpu} math 12000 arxiv ce
echo "Training Math BERT.."
python3 bert_train.py 1 ${gpu} 2 math 1 arxiv ce
echo "***********************************************************************"
