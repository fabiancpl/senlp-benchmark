# Tasks included in SELU

This is the inventory of SELU tasks and their respective datasets used for model evaluation.

The purpose of the notebook inside each folder is exploring the dataset and apply simple transformations such as removing useless columns and normalizing column names to make standard further preprocessing processes. Some basic exploration and data quality analysis are carried out to identify warnings such as data duplication or label imbalance. At the end of each script, a new version of the dataset is stored in parquet or pickle format.

Note that the *requirement_completion* task has been redefined to fill only POS verbs, thus the corresponding tokenization and POS tagging processes are also carried out.

|   **Category**  |       **Task ID**        |                        **Task definition**                        | **Instances** | **Targets** |
|:---------------:|:------------------------:|:-----------------------------------------------------------------:|:-------------:|:-----------:|
|    **Binary**   |        _bug_issue_       | Is the issue reporting a bug?                                     |     38,219    |      2      |
|                 | _functional_requirement_ | IDoes the requirement include some functional aspect?             |      956      |      2      |
|                 |       _incivility_       | Does the text show unnecessary rude behavior?                     |     1,546     |      2      |
|                 |  _quality_requirement_   | Does the requirement include some quality aspect?                 |      956      |      2      |
|                 |      _safety_issue_      | Is the issue reporting safety-related concerns?                   |     1,916     |      2      |
|                 |  _security_requirement_  | Does the requirement include some security aspect?                |      510      |      2      |
|                 |      _tone_bearing_      | Does the text have an unnecessarily disrespectful tone?           |     6,597     |      2      |
| **Multi-class** |    _closed_question_     | Which is the reason for closing the question after moderation?    |    140,272    |      5      |
|                 |    _commit_intention_    | Is the commit perfecting or correcting the code?                  |     2,533     |      3      |
|                 |    _issue_intention_     | Which is the intention expressed in the issue?                    |     6,375     |      7      |
|                 |      _issue_type_        | Is the issue related to a bug, an enhancement or a question?      |    803,417    |      3      |
|                 |    _question_quality_    | Is the question of good quality or does it require moderation?    |     60,000    |      3      |
|                 |      _review_type_       | Which is the intention expressed in the app review?               |     1,390     |      4      |
|                 |       _sentiment_        | Which is the sentiment expressed in the text?                     |     13,144    |      3      |
| **Multi-label** |   _comment_type_java_    | Which kind of contents are detailed in the code comment? (Java)   |     9,339     |      7      |
|                 |   _comment_type_pharo_   | Which kind of contents are detailed in the code comment? (Pharo)  |     1,587     |      7      |
|                 |  _comment_type_python_   | Which kind of contents are detailed in the code comment? (Python) |     2,290     |      5      |
|                 |     _review_aspect_      | Which aspect is involved in the API review?                       |     4,522     |      11     |
|                 |       _smell_doc_        | Does the API documentation smell?                                 |     1,000     |      5      |
|  **Regression** |      _story_points_      | What is the effort estimated for the development task?            |     23,313    |    [1–96]   |
|     **NER**     |      _se_entities_       | What kind of SE terminology is discussed in the text?             |     2,718     |      20     |
|     **MLM**     | _requirement_completion_ | How to fill the specification with the proper user action?        |      40*      |      *      |
