# Paper: Exploiting Personal Characteristics of Debaters for Predicting Persuasiveness

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3778298.svg)](https://doi.org/10.5281/zenodo.3778298)

This is the code for the paper *Exploiting Personal Characteristics of Debaters for Predicting Persuasiveness*.

Khalid Al-Khatib, Michael Völske, Shahbaz Syed, Nikolay Kolyada, and Benno Stein

```
@InProceedings{stein:2020m,
  author =              {Khalid Al-Khatib and Michael V\"olske and Shahbaz Syed and Nikolay Kolyada and Benno Stein},
  booktitle =           {58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)},
  month =               jul,
  pages =               {7067-7072},
  publisher =           {Association for Computational Linguistics},
  site =                {Seattle, USA},
  title =               {{Exploiting Personal Characteristics of Debaters for Predicting Persuasiveness}},
  url =                 {https://www.aclweb.org/anthology/2020.acl-main.632},
  year =                2020
}
```

The provided code contains the set of PySpark jobs used for preprocessing and construction of final datasets from [Reddit Crawl](https://files.pushshift.io/reddit/) corpus. The jobs under `pyspark/` folder have to be packed into a zip archive and run with commands listed in `run.sh`. Respective build commands in the Makefile can be used.

The provided Jupyter notebooks demostrate the main steps taken for two tasks studied in the paper: load the preprocessed datasets, prepare feature dictionaries, train models and do feature selection. (Task 1: notebooks 1-3, Task 2: notebook 4).
