import logging

"""
## 日志事件级别
级别排序：CRITICAL > ERROR > WARNING > INFO > DEBUG
CRITICAL # 严重的错误，表示程序已经不能继续执行。
ERROR # 由于严重的问题，程序的某些功能已经不能正常执行。
WARNING # 表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
INFO # 确认程序按预期进行
DEBUG # 细节信息，仅当诊断问题时适用。
默认级别WARNING，意味着只会追踪该级别及以上的事件，除非更改日志配置。

## Handler处理器，将日志发送到不同地方
常见的处理器
StreamHandler  # 屏幕输出
FileHandler  # 文件记录
BaseRotatingHandler  # 标准的分割日志文件
RotatingFileHandler  # 按文件大小记录日志
TimeRotatingFileHandler  # 按时间记录日志

## 控制台输出参数
name %(name)s 日志的名称
asctime %(asctime)s 可读时间
......

"""


def get_logger(name='root'):
    # asctime：字符串形式的当前时间， levelname：文本形式的日志级别， message：用户输出的消息
    # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 能够将日志信息输出到sys.stdout,sys.stderr或者类文件对象（更确切，就是能够支持write()和flush()方法的对象）
    handler = logging.StreamHandler()  # 控制台输出日志
    handler.setFormatter(formatter)  # 设置控制台打印格式

    logger = logging.getLogger(name)  # 创建logger，默认root
    logger.setLevel(logging.INFO)  # 设置日志记录级别
    logger.addHandler(handler)  # 将日志内容传递到相关联的handler中
    # print(handler)
    return logger
