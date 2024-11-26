import logging
import time
import os

def log_config(name='logging'):
    if not os.path.exists('./log'):
        os.mkdir('./log')

    logdatetime = time.strftime("%Y-%m-%d-%H:%M:%S--")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")    
    fh = logging.FileHandler("C:\\Users\\AlarmTran\\Meng2018Largescale\\log\\cellnet_train_valid.log")
    # print(fh)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # if os.path.exists('./log/current--' + name + '.log'):
    #     os.remove('./log/current--' + name + '.log')
    # fhc = logging.FileHandler('./log/current--' + name + '.log')
    # fhc.setLevel(logging.INFO)
    # fhc.setFormatter(formatter)
    # logger.addHandler(fhc)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return

