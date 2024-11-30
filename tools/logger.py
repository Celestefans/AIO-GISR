import logging
import sys

def get_logger(filename, mode='w'):
    """
    Return a logger instance that writes in filename
    Args:
        filename: (string) path to log.txt
        mode: 'w' for overwrite, 'a' for append
    """
    logger = logging.getLogger('Logger')
    logger.setLevel(logging.INFO)
    
    # Remove all existing handlers
    logger.handlers = []
    
    # logging to file
    handler = logging.FileHandler(filename, mode=mode)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    # logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    
    return logger