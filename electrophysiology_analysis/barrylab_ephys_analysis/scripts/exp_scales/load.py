
import os

from barrylab_ephys_analysis.recording_io import Recordings


def find_recording_session_groups(fpath):
    """Returns paths to all sub-folders. If run on animal directory, should yield recording day folders.
    """
    return [os.path.join(fpath, session_date_folder) for session_date_folder in sorted(os.listdir(fpath))]


def find_recording_sessions(fpath, experiment_file_name='experiment_1.nwb'):
    """Returns paths to all sub-folders containing file with experiment_file_name.
    """
    recording_folders = []
    for subfolder in os.listdir(fpath):
        potential_recording_folder_path = os.path.join(fpath, subfolder)
        if experiment_file_name in os.listdir(potential_recording_folder_path):
            recording_folders.append(potential_recording_folder_path)

    return recording_folders


def get_paths_to_animal_recordings_on_single_day(fpath, animal_id):
    """Returns the paths to all recordings found in the sub-folder specified by animal_id

    :param str fpath: path to ExpScales folder that contains folders with animal_ids
    :param str animal_id: fpath sub-folder name containing the data of the animal
    :return:
    """
    return sorted(find_recording_sessions(find_recording_session_groups(os.path.join(fpath, animal_id))[0]))


def load_recordings_of_one_animal_on_single_day(fpath, animal_id, *args, **kwargs):
    """Returns :py:class:`recording_io.Recordings` instance for all recordings of the animal_id
    on the first recording day found in the animal sub-folder

    :param str fpath: path to ExpScales folder that contains folders with animal_ids
    :param str animal_id: fpath sub-folder name containing the data of the animal
    :param args: passed on to :py:func:`recording_io.Recordings`
    :param kwargs: passed on to :py:func:`recording_io.Recordings`
    :return:
    """
    return Recordings(get_paths_to_animal_recordings_on_single_day(fpath, animal_id), *args, **kwargs)


def load_recordings_of_all_animals(fpath, animal_ids, *args, **kwargs):
    """Returns list of :py:class:`recording_io.Recordings` classes instantiated
    with paths to recordings of specified animals.

    :param str fpath: path to ExpScales folder that contains folders with animal_ids
    :param tuple animal_ids: list of sub-folder names specifying data for each animal
    :param args: passed on to :py:func:`recording_io.Recordings`
    :param kwargs: passed on to :py:func:`recording_io.Recordings`
    :return:
    """
    return [load_recordings_of_one_animal_on_single_day(fpath, animal_id, *args, **kwargs)
            for animal_id in animal_ids]
