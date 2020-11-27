# Relationships Between Countries United Nation General Assembly Speeches

The aim of this project is to identify relationships between the topics mentioned in a countries United Nations General Debate speech, and three main factors:

- The economical and social development of a country.
- The geographical location of a country.
- The year the speech was given.

This is achieved by vectorising the corpus of speeches given from 1970 to 2015, by calculating the Term Frequency-inverse Document Frequency of each speech. Latent semantic analysis (LSA) is then applied to these vectors, in order to generate topics, and reduce the dimensionality of the data. Principle component analysis (PCA) is also applied to simply highlight the similarities between the two methods.

A full explanation of the processes used, as well as analysis of obtained results, can be found in the corresponding  [report document](Report/Relationships_Between_Countries_United_Nation_General_Assembly_Speeches.pdf).

This project was submitted as part of my MSc Advanced Computer Science degree and received a grade of Distinction.

## Prerequisites

This project was built using Python 3.8, and so is only guaranteed to work with 3.8 and above. However, any version of Python 3 should run the code. The following libraries are also required:

- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)
- [Sklearn](https://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org)

## How To Use

_Note: This project was built for research purposes, and provides an interactive shell to manipulate the data. Therefore, using the project requires an understanding of the variables and functions._

Before running the project unzip the [data_sets.zip](data_sets.zip) file. The unzipped directory should be named `data_sets` and be located in the same directory as the [preprocessing.py](preprocessing.py) file.

This project is interacted through the python `code.interact` module. To run the program navigate your console to the project directory and run the command:

```
python preprocessing.py
```

The program will read the data set, and carry out all preprocessing required. This process will create a directory called generate and will write files to this directory. Once this is completed the python shell will load. This shell allows for all defined functions in the program to be called in any order. Below is a brief description of the main functions and variables required to interact with the data/results:

- `data` : Data-frame containing the entire data set processed via LSA.
- `data_2015` :  Data-frame containing the 2015 data set processed via LSA.
- `pdf_2015` : Data-frame containing the entire data set processed via PCA.
- `pdf_2015` :  Data-frame containing the 2015 data set processed via PCA.
- `hdi_tag` : Variable containing human development index tagging information for entire data set visualisation.
- `continent_tag` : Variable containing continent tagging information for entire data set visualisation.
- `decade_tag` : Variable containing decade tagging information for entire data set visualisation.
- `hdi_tag_2015` : Variable containing human development index tagging information for 2015 data set visualisation.
- `continent_tag_2015` : Variable containing continent tagging information for 2015 data set visualisation.
- `plot_2d(x, y, tagging, x_label, y_label)` : Produces a 2D plot of provided data. Colours data points based on the provided tagging variable.
- `plot_3d(x, y, z, tagging, x_label, y_label, z_label)` : Produces a 2D plot of provided data. Colours data points based on the provided tagging variable.

To run the given implementation of LSA or PCA on another piece of data, the following functions can be called:

- `principle_component_analysis(data)` : Runs PCA on a given preprocessed data set. Returns data frame of processed data.
- `latent_semantic_analysis(data, tfidf, tfidf_vec)` : Runs LSA on a given preprocessed data set. Returns data-frame and LSA object.

## License

This project is licensed under the terms of the [Creative Commons Attribution 4.0 International Public license](License.md)