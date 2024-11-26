### Requirements
* notebook==5.7.4
* numpy==1.15.4
* Pillow==5.3.0
* scikit-image==0.14.1
* scikit-learn==0.20.1
* scipy==1.1.0
* opencv>=2.4.12
* tensorboard>=1.4.0
* tensorflow>=1.4.0
* tqdm==4.28.1

### Dataset
Tải dataset và test tại [https://github.com/monaen/Meng2018Largescale/tree/data](https://github.com/monaen/Meng2018Largescale/tree/data) - Tải bản Augmented


### Cách tải
1. clone the code branch of the project (Notice: we do not suggest to clone the entire project)
```commandline
git clone https://github.com/monaen/Meng2018Largescale.git --branch code --single-branch
```

2. head to the `data` folder and download the data
```commandline
cd Meng2018Largescale/data
./download_augmented_Train_Test_Valid.sh
```

3. train (test) the model
```commandline
python main.py --dataset "data/Augmented" --batchsize 2000 --imgsize 128
```

4. run the demo
```commandline
python demo.py
```

Code này đã được update và fix bug
