# 添加日志记录
import logging

def setup_logger(exp_path, name):
    
    # 创建logger，设置级别
    logger = logging.getLogger(exp_path)
    logger.setLevel(logging.INFO)

    # 创建文件handler
    handler = logging.FileHandler(f'{exp_path}/{name}.log', mode='a')
    handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(handler)
    return logger