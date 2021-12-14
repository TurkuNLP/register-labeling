# register-labeling

Scripts used to label OSCAR with register information

Register tagged data can be found at https://huggingface.co/datasets/mhtoin/register_oscar

Data should be added to the data directory. By default, the expected structure is data/two-letter-language-code/1/ 2/ 3/

The model can be downloaded from  http://dl.turkunlp.org/register-labeling-model/

## Data setup

To download a specific language from the OSCAR dataset, run

```python download_dataset.py --lang {your_language_code}```

This creates the data directory (if missing) and downloads the dataset as .csv, by default with the 'id' and 'text' columns. 

To convert the downloaded csv to tsv-format which the classifier supports out of the box, run

``` python convert_to_tsv.py --lang {language_code}```

Finally, split the data into parts for parallelisation with:

``` sh split_dataset {language_code}```

## Labeling

Once data has been downloaded and split appropriately, labeling can be run (in a slurm environment) with:

```sh submit_array.sh {language_code}```

The script monitors your job queue and submits a new batch array job whenever there is room, until the whole dataset is done. You need the script to use the correct user's queue (line 5):

```count=$(squeue -u {your_username} | egrep -c 'predict')```

The script uses the 'num' and 'max' parameters to keep track of its progress. Num is the 'current package' (starting from 1) and max is number of packages overall. The script runs while num < max. Modify these parameters according to your dataset

Tagging can also be run on a single package (specify array size in predict_array.sh):

```sbatch predict_array.sh en models/xlmrL.h5 test.log {language_code} {package_number}```

To simply run the evaluation script without a slurm environment:
```
python predict_labels.py \
  --model_name jplu/tf-xlm-roberta-large \
  --load_weights "models/xlmrL.h5" \
  --load_labels "models/labels.txt" \
  --test data/test.tsv \
  --bg_sample_rate 1.0 \
  --input_format tsv \
  --threshold 0.5 \
  --seq_len 512 \
  --batch_size 7 \
  --epochs 0 \
  --test_log_file "test.lg" \
  --save_predictions "output/preds."
  ```
  
 ## Combining results
 
 Once the tagging has been done, the final tagged dataset can be constructed by mapping the predictions to the label name and combining label names with prediction probabilities and the actual text for each document. This can be accomplished with
 
 ``` sh construct_dataset.sh {language_code}```

There is also a script for running the construction in the slurm environment (in case of a particularly large dataset):

```sbatch slurm_construct.sbatch {language_code}```

Cite:
Rönnqvist, S., Skantsi, V., Oinonen, M., & Laippala, V. (2021). Multilingual and Zero-Shot is Closing in on Monolingual Web Register Classification. Nordic Conference on Computational Linguistics, Linköping Electronic Conference Proceedings. https://aclanthology.org/2021.nodalida-main.16/

Repo, Liina. Skantsi, Valtteri, Rönnqvist, Samuel, Hellström, Saara, Oinonen, Miika, Salmela, Anna, Biber, Douglas, Egbert, Jesse, Pyysalo, Sampo & Laippala, Veronika (2021). Beyond the English Web: Zero-Shot Cross-Lingual and Lightweight Monolingual Classification of Registers. Proceedings of the Conference of the European Chapter of Association for Computational Linguistics: Student Workshop.
