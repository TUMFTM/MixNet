import os


def list_dirs_with_file(root_dirs, file_name):
    """Returns a list of all dirs under any of the dirs in root_dirs which contain
    a file which's name contains the the string give in file_name.

    args:
        root_dirs: (list), list of directories to search under.
        file_name: (str), the phrase to search for in the file names.

    returns:
        A list with all the dirs which contain a file which includes the file_name string.
    """

    assert isinstance(root_dirs, list), "The given root_dirs is not a list."

    dir_list = []

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            continue

        for path, _, files in os.walk(root_dir):
            for name in files:
                if file_name in name:
                    dir_list.append(os.path.dirname(os.path.join(path, name)))

    return list(set(dir_list))


def list_files_with_name(roots, file_name=""):
    """Lists every file under the given roots which contain the file_name.
    Some of the paths in roots can also be a file name it self, these are just
    simply going to be added to the list.

    args:
        roots: (list), the list of dirs and files to search under.
        file_name: (str), the string that every returned file name must contain.

    returns:
        the list of files.
    """

    assert isinstance(roots, list), "The given root_dirs is not a list."

    file_list = []
    root_dirs = []

    for path in roots:
        if os.path.exists(path):
            if os.path.isfile(path) and (file_name in path):
                file_list.append(path)
            elif os.path.isdir(path):
                root_dirs.append(path)

    dir_list = list_dirs_with_file(root_dirs, file_name)

    for dir in dir_list:
        for path in os.listdir(dir):
            total_path = os.path.join(dir, path)
            if os.path.isfile(total_path) and (file_name in total_path):
                file_list.append(total_path)

    return list(set(file_list))
