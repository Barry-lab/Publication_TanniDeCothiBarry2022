# Electrophysiology Analysis

## Environment

Install `conda`: https://docs.conda.io/en/latest/miniconda.html

Create Python version 3.7.3 environment

    conda create -n blea python=3.7.3

Enter the new conda environment

    conda activate blea    

Install required packages using `pip` and the `requirements.txt` in the same directory as this README.

    pip install -r requirements.txt

Install conda build

    conda install conda-build

Add this directory to python path using conda build

    conda develop /absolute/path/to/the/directory/of/this/readme

## Producing figures

Run preprocessing script preferably on a powerful machine with plenty of memory. This is a long process. The input to this script should be the path to the raw data directory - containing a sub-folder for each animal.

    python barrylab_ephys_analysis/scripts/exp_scales/paper_preprocess.py /path/to/raw/data/root/directory

Now run the script for creating the figures

    python barrylab_ephys_analysis/scripts/exp_scales/paper_figures.py /path/to/raw/data/root/directory

The figures will be written to `/path/to/raw/data/root/directory/Analysis/PaperFigures` directory.

Population vector change rate values needed for comparative analysis with visual similarity will be written to `/path/to/raw/data/root/directory/Analysis/df_population_vector_change.p`
