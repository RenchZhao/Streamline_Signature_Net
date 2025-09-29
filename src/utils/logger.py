import logging
import time
import os
import sys

import random

# def create_logger(final_output_path, description=None):
#     if description is None:
#         log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
#     else:
#         log_file = '{}_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'), description)
#     head = '%(asctime)-15s %(message)s'
#     logging.basicConfig(filename=os.path.join(final_output_path, log_file),
#                         format=head)
#     clogger = logging.getLogger()
#     clogger.setLevel(logging.INFO)
#     # add handler
#     # print to stdout and log file
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
#     clogger.addHandler(ch)
#     return clogger

def create_logger(final_output_path, description=None):
    # 生成唯一的日志器名称（基于时间和描述）
    logger_name = time.strftime('%Y-%m-%d-%H-%M')
    if description:
        logger_name += f"_{description}"
    # time.sleep(random.uniform(0, 1))
    # 获取具有唯一名称的日志器，而不是根日志器
    logger = logging.getLogger(logger_name+'{:.2f}'.format(random.uniform(0, 1))) #短时多个日志防止冲撞
    
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建并配置文件处理器
    if description is None:
        log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_file = '{}_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'), description)
    
    file_handler = logging.FileHandler(os.path.join(final_output_path, log_file))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)-15s %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 创建并配置流处理器（输出到控制台）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    
    return logger