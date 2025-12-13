import logging
import colorlog

def log_init():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if log.hasHandlers():
        log.handlers.clear()
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] [%(levelname)s] - %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    log.addHandler(handler)
    
    return log