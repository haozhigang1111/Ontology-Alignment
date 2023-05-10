# Siamese-GCN

This folder includes the implementation of Siamese-GCN introduced in the paper ****Ontology Alignment with Semantic and Structural Embeddings****.
The HeLis and FoodOn ontologies, the OAEI Conference track, the OAEI Anatomy track, the OAEI largebio track task1 and our food classification ontologies, which are adopted for the evaluation in the paper, are under **owl_data/xx/**. Their **GS_file**s and **Reference_file** are under **owl_align**.
Note food classification ontologies adopted have been pre-processed by translating their Chinese labels into English lables and dropping theri instances .


### Dependence 
Our codes in this package require: 
  1. Python 3.7.9
  2. torch 1.7.0
  3. gensim 4.0.1
  4. OWLready2 0.29
  5. nltk 3.6.7
  6. bert-serving 1.10.0
  7. numpy 1.21.4   
  8. [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star)
  9. [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher)
  10. [AML](https://github.com/AgreementMakerLight/AML-Project)
  11. [RDGCN](https://github.com/StephanieWyt/RDGCN)
  12. [LogMap-ML](https://github.com/KRR-Oxford/OntoAlign/tree/main/LogMap-ML)

### Startup

### Pre-process #1: Transform the original OWL files into the format required for the experiments.
Navigate to the **data/** directory and then run:

``` python owl2id --left_onto_file ../owl_data/xx/xx.owl --right_onto_file ../owl_data/xx/xx.owl --output dir ```

This will generate a corresponding file directory **dir/** in the **data/** folder, which includes four files: 'ent_ids_1' and 'ent_ids_2' record the IDs corresponding to the two ontology concepts, and 'triples_1' and 'triples_2' record the relationships between the two ontology concepts.


### Pre-process #2: Class Name and Path Extraction.
Navigate to the **class&path/** directory and then run:

``python name_path.py --onto_file ../owl_data/xx.owl --name_file xx_class_name.json --path_file xx_all_paths.txt``

This is to extract the name information and path information for each class in an ontology. 
It should be executed separately for the to-be-aligned ontologies.

### Pre-process #3: Embedding Models.

You can use the ontology tailored [OWL2Vec\* embedding](https://github.com/KRR-Oxford/OWL2Vec-Star). Download OWL2Vec\*, run:

```python OWL2Vec_Standalone_Multi.py --ontology_dir tmp_owl/ --embedding_dir owl_embedding/xx_emb```

The ``ontology_dir`` is the directory where the two ontologies to be aligned are stored, and the ``embedding_dir`` is the output directory. You can also directly run by setting the parameters in ``default_multi.cfg``:

```python OWL2Vec_Standalone_Multi.py --config_file default_multi.cfg```

### Pre-process #4: Transform the Embedding files into the format required for the experiments.

Navigate to the **data/** directory and then run:

```python create_embeddings --owl2vec_file ../owl_embedding/xx_emb --left_owl2id xx/ent_ids_1 --right_owl2id xx/ent_ids_2 --output xx/embeddings  --emb_type owl2vec```

Additionally, you can modify the  ``emb_type`` parameter to generate BERT or random embeddings. For example:

```python create_embeddings --owl2vec_file ../owl_embedding/xx_emb --left_owl2id xx/ent_ids_1 --right_owl2id xx/ent_ids_2 --output xx/embeddings_bert  --emb_type bert```

```python create_embeddings --owl2vec_file ../owl_embedding/xx_emb --left_owl2id xx/ent_ids_1 --right_owl2id xx/ent_ids_2 --output xx/embeddings_random  --emb_type random```

When you want to generate BERT embeddings, remember to start the BERT service in the console:

``` bert-serving-start -model_dir uncased_L-12_H-768_A-12```
[Here is the download link for BERT pre-trained models](https://github.com/google-research/bert)


### Pre-process #5: Run the original system
Download [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher), build by Maven, run:

```java -jar target/logmap-matcher-4.0.jar MATCHER file:/xx/helis_v1.00.owl file:/xx/foodon-merged.owl logmap_output/ true```

This leads to LogMap output mappings, overlapping mappings and anchor mappings. 

You also download [AML](https://github.com/AgreementMakerLight/AML-Project), and directly run the  ``AgreementMakerLight.jar``. This will leads to AML outputs.


### Step #1: Sample
Navigate to the **class&path/** directory.
For FoodOn-HeLis and FoodClassificaiton ontologies, run:

```python sample.py --anchor_mapping_file logmap_output/logmap_anchors.txt --left_class_name_file xx_class_name.json --left_path_file xx_all_paths.txt --right_class_name_file xx_class_name.json --right_path_file xx_all_paths.txt --train_file ../data/xx/mappings_train.txt --valid_file ../data/xx/mappings_valid.txt```
For the OAEI tracks, you need set ``anchor_mapping_file`` to ``logmap_output/logmap2_mappings.txt`` (or use AML outputs).

It outputs mappings_train.txt and mappings_valid.txt.
The branch conflicts which are manually set for higher quality seed mappings are set inside the program via the variable ``branch_conflicts``.
See "help" and comments inside the program for more settings. 

### Step #2: Train, Valid and Predict

```python run.py  --train_path_file data/xx/mappings_train.txt --valid_path_file data/xx/mappings_valid.txt --left_w2v_dir owl_embedding/xx_emb --right_w2v_dir owl_embedding/xx_emb --embedding_type owl2vec --nn_dir checkpoints/xx/```

```python predict_mappings.py  --left_path_file class&path/xx_all_paths.txt --right_path_file class&path/xx_all_paths.txt --left_class_name_file class&path/xx_class_name.json --right_class_name_file  class&path/xx_class_name.json  --candidate_file candidate_prediction/xx.txt  --prediction_out_file outputs/xx.txt```

Note the candidate mappings should be pre-extracted, usually with a high recall. We use the overlapping mappings by LogMap for FoodOn-HeLis. For the other candidate mappings, you can run ``create_predicted_mappings.py`` in **candidate_prediction/**.


### Step #3: Evaluate 

#### With approximation
Calculate the recall w.r.t. the GS, and sample a number of mappings for annotation, by:

```python evaluate_for_approximate.py --threshold 0.5 --anchor_file logmap_output/logmap_anchors.txt```

It will output a file with a part of the mappings for human annotation. 
The annotation is done by appending "true" or "false" to each mapping (see annotation example in evaluate.py).
With the manual annotation and the GS, the precision and recall can be approximated by:

```python approximate_precision_recall.py (or approximate_precision_recall_gb.py)```

Please see Eq. (13)(14)(15) in the paper for how the precision and recall approximation works.
For more accurate approximate, it is suggested to annotate and use the mappings of other systems to approximate the GS. 
Besides the original LogMap and Siamese-GCN, you can also consider [AML](https://github.com/AgreementMakerLight/AML-Project) ,[RDGCN](https://github.com/StephanieWyt/RDGCN), [LogMap-ML](https://github.com/KRR-Oxford/OntoAlign/tree/main/LogMap-ML).

#### Straightforward 
If it is assumed that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```python evaluate_straightforward.py --prediction_out_file outputs/xx.txt --oaei_GS_file owl_align/xx.rdf --left_class_file rdf2rdf/xx_class.list --right_class_file rdf2rdf/xx_class.list```

========================================

#### Supplement



