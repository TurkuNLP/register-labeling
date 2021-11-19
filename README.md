# register-labeling

Scripts used to label OSCAR with register information

Register tagged data can be found at https://huggingface.co/datasets/mhtoin/register_oscar

Data should be added to the data directory. By default, the expected structure is data/two-letter-language-code/1/ 2/ 3/

The model can be downloaded from  http://dl.turkunlp.org/register-labeling-model/

To run tagging on a whole language package in a slurm environment (split into further packages of 200 parts/package):

```sh submit_array.sh {language_code}```

To run tagging on a single package (specify array size in predict_array.sh):

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

Cite:
Rönnqvist, S., Skantsi, V., Oinonen, M., & Laippala, V. (2021). Multilingual and Zero-Shot is Closing in on Monolingual Web Register Classification. Nordic Conference on Computational Linguistics, Linköping Electronic Conference Proceedings. https://aclanthology.org/2021.nodalida-main.16/


