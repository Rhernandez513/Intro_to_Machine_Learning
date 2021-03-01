Homework 1: Setup and Prerequisites
UIC CS 412, Spring 2021

Create a directory for CS 412 homework (e.g., cs412-hw)

Move hw1.zip to that directory and unzip it.

Install anaconda for python 3.8: https://www.anaconda.com/download
If you already have anaconda, make sure it's updated to this version.

In your terminal, check whether anaconda was installed:

elena-macbook:cs412-hw elena$ conda --version

If the command cannot be found, then make sure you have closed and re-opened the terminal window after installing it.

elena-macbook:cs412-hw elena$ python --version
Python 3.8.5 :: Anaconda, Inc.

Create a python 3.8 environment:

elena-macbook:cs412-hw elena$ conda create -n cs412 python=3.8 anaconda

Activate the environment:

elena-macbook:cs412-hw elena$ source activate cs412   #omit the source part on windows

Change to homework directory:

(cs412) elena-macbook:cs412-hw elena$ cd hw1

Start the Jupyter notebook:

(cs412) elena-macbook:hw1 elena$ jupyter notebook &

This should open a notebook in your web browser and display the contents of the current directory. Select hw1.ipynb, and follow the instructions in the notebook.



