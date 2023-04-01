import logging
import traceback

def get_logger(log_file_path: str, task_name: str) -> logging.Logger:
    """
    Returns a logger object with handlers to log messages to both the console and a specified log file.

    Args:
        log_file_path (str): The file path to write the log messages to.
        task_name (str): The name of the task to include in the log messages.

    Returns:
        logging.Logger: A logger object with the specified handlers.
    """
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    log_file_handler = logging.FileHandler(log_file_path, mode="w") # <--- mode="w" to overwrite the log file; "a" to append
    log_file_handler.setLevel(logging.INFO)
    log_file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_file_handler)

    return logger


def log_error(message: str, error: Exception, error_fpath: str) -> None:
    """
    Log an error message and write the error to a specified file
    Args:
        message (str): The error message.
        error (Exception): The exception instance.
        error_fpath (str, optional): The file path to write the error message to. Defaults to None.
    """
    with open(error_fpath, 'a', encoding='utf-8') as f:
        err_msg = f"{message} Error: {str(error)}"
        f.write(err_msg)
        f.write('\n')
        traceback.print_exc(file=f)