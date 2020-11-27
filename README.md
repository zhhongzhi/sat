# sat
The accompanying code for "[Table Fact Verification with Structure-Aware Transformer](https://www.aclweb.org/anthology/2020.emnlp-main.126/)" (short paper of EMNLP 2020).

1.Enveriment

	python 3.7
	pip install sklearn   
	pip install torch==1.1   
	pip install pytorch-pretrained-bert==0.6.2   
	pip install transformers==2.6   
	bert-base-uncased/  download and unzip the pre-trained bert here. (the multilingual bert is prefered)
2.train from scratch

    a.clone FacTable via: git clone https://github.com/wenhuchen/Table-Fact-Checking.git
    b.cp sat_main.py Table-Fact-Checking-master
    c.mkdir models
    d.train the model: 
        python sat_main.py # with default configuration (vertical scan)
        It takes about 50 mins on a single V100 to run an epoch, and takes 15-18 epochs to achieve best validataion results.
    
    e. All the ablation experiment is runnable. 
       e.g: python sat_main.py --use_vm no # ablation on attention masking
       please refer to sat_main.py for more configuration details. 

Please raise an issue if any questions. 


