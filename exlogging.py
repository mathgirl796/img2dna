import logging
import os
import datetime

def create_logger():
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 创建文件处理器
    log_folder = 'log'
    os.makedirs(log_folder, exist_ok=True)  # 创建log文件夹，如果已存在则跳过
    log_filename = datetime.datetime.now().strftime(f"%Y-%m-%d-%H-%M-%S_pid{os.getpid()}.txt")
    log_filepath = os.path.join(log_folder, log_filename)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 将处理器添加到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger