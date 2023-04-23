import logging


def get_logger(name='root'):
    # asctime：字符串形式的当前时间， levelname：文本形式的日志级别， message：用户输出的消息
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 能够将日志信息输出到sys.stdout,sys.stderr或者类文件对象（更确切，就是能够支持write()和flush()方法的对象）
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # print(handler)
    return logger
