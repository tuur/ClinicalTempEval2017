# Code

This directory contains the code of the [KULeuven-LIIR submission](http://aclweb.org/anthology/S/S17/S17-2181.pdf) for [Clinical TempEval 2017](https://www.aclweb.org/anthology/S/S17/S17-2093.pdf), where it participated in all six subtasks, using a combination of Support Vector Machines (SVM) for event and temporal expression detection, and a structured perceptron for extracting temporal relations, and performed among the top-ranked systems.

> In case of usage, please cite the corresponding publications (see bottom of this page).

#### What it does
Training and prediction for the following six temporal information extraction tasks:
1. Detection of event spans (ES)
2. Identification of event attributes (EA)
3. Detection of temporal expressions (TS)
4. Attribute identification of temporal expressions
(TA)
5. Extraction of document-creation-time relations
for events (DR)
6. Extraction of narrative container relations (CR)

> Any questions? Feel free to send me an email at aleeuw15@umcutrecht.nl

### Requirements
* [Gurobi](https://www.gurobi.com)  (create account, download gurobi, and run setup.py)
* [Python2.7](https://www.python.org/downloads/release/python-2711/)
  * [Argparse](https://pypi.python.org/pypi/argparse)
  * [Numpy](http://www.numpy.org/)
  * [SciPy](https://www.scipy.org/)
  * [Networkx](https://networkx.github.io)
  * [Scikit-Learn](http://scikit-learn.org/stable/)
  * [Pandas](http://pandas.pydata.org/)
* [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml)

It uses the [Stanford POS tagger](http://nlp.stanford.edu/software/tagger.shtml) for POS features. For this reason it is required to have the Stanford POS Tagger folder (e.g. `stanford-postagger-2015-12-09`), the `stanford-postagger.jar`, and the `english-bidirectional-distsim.tagger` file at the same level as `run_pipeline_on_mini_silver_data.sh`.

### Running the code
To use the pipeline that trains the models, and provides predictions on raw test texts you can run the script:
```
sh run_pipeline_on_mini_silver_data.sh | tee log.txt
```
It will output the trained models and predicted anafora xmls to the directory `output`.

To get more detailed info on setting hyperparameters you can inspect `run_pipeline_on_mini_silver_data.sh` or run `python main_entities.py -h` or `python main_rels.py -h` for more elaborate options.

By default this script will just use a very small dataset `mini_silver_data` of small case study reports obtained from the web, that were machine-annotated so are very faulty and act just to provide a working example of the code.

### Data

In the papers we used the [THYME](https://clear.colorado.edu/TemporalWiki/index.php/Main_Page) corpus as used for the [Clinical TempEval](http://alt.qcri.org/semeval2016/task12/index.php?id=data) shared task (2015, 2016 or 2017). So, training or test data should be provided in the anafora xml format, in a folder structure as indicated below (or as in the `mini_silver_data`).

* `Train`
  * `ID001_clinic_001`
    * `ID001_clinic_001`
    * `ID001_clinic_001.Temporal-Relation.gold.completed.xml`
  * ...

And similarly for the test data (but possibly without the anafora .xml).

##### Optional: cTAKES Features
If you want to use POS and dependency features from the cTAKES clinical pipeline you need to provide the cTAKES output xml files as well (for train and test). The folder structure of theses directories is:

`CTAKES_XML_DIR`
* `ID001_clinic_001.xml`
* ...

You can specify the ctakes xml directories for train and test at the top of the `run_pipeline_on_mini_silver_data.sh` script.
## References

```
@InProceedings{leeuwenberg-moens:2017:SemEval,
  author    = {Leeuwenberg, Artuur  and  Moens, Marie-Francine},
  title     = {KULeuven-LIIR at SemEval-2017 Task 12: Cross-Domain Temporal Information Extraction from Clinical Records},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {1030--1034},
}

@InProceedings{leeuwenberg2017structured:EACL,
  author    = {Leeuwenberg, Artuur and Moens, Marie-Francine},
  title     = {Structured Learning for Temporal Relation Extraction from Clinical Records},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
}
```
