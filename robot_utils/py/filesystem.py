import os
import glob


def get_ordered_subdirs(dir_name: str, pattern=[''], recursive=False, depth="/**/*"):
    """
    Args:
        dir_name (str): the directory name to look for sub-directories
        pattern (List(str)): the pattern to follow when filtering the sub-directory name
        recursive (bool): whether to search recursively
        depth (str): specify the depth to look for files

    Returns:
        A list of oredered sub-directories in the dir_name that fullfils the pattern

    """
    return sorted( filter( lambda x: os.path.isdir(x) and all([p in x for p in pattern]),
                                    glob.glob(dir_name + depth, recursive=recursive)))


def get_ordered_files(dir_name: str, pattern=[''], recursive=False, depth="/**/*"):
    """
    Args:
        dir_name (str): the directory name to look for sub-directories
        pattern (List(str)): the pattern to follow when filtering the sub-directory name
        recursive (bool): whether to search recursively
        depth (str): specify the depth to look for files

    Returns:
        A list of oredered filenames in the dir_name that fullfils the pattern

    """
    return sorted( filter( lambda x: os.path.isfile(x) and all([p in x for p in pattern]),
                           glob.glob(dir_name + depth, recursive=recursive) ) )
