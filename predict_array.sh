#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -p gpu
#SBATCH -t 3:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH --output=arr_logs/array_%A-%a.out    # Standard output and error log
#SBATCH --array=0-3

echo "START: $(date)"
mkdir -p arr_logs

if [ -f models/*.gz ];
  then gzip -d models/*.gz
fi

mkdir -p arr_logs

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load tensorflow/2.2-hvd
source transformers3.4/bin/activate

SRC=fr
TRG=$1

export TEST=data/$4/$5

mkdir -p output/$4/$5

MODEL_ALIAS="xlmR"
MODEL="jplu/tf-xlm-roberta-large"

BS=7
BGrate=1.0

LABELS="models/labels.txt"

echo "Settings: src=$SRC trg=$TRG bg=$BG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"

srun python predict_labels.py \
  --model_name $MODEL \
  --load_weights "$2" \
  --load_labels "$LABELS" \
  --test $TEST/$4.tsv.$SLURM_ARRAY_TASK_ID \
  --bg_sample_rate $BGrate \
  --input_format tsv \
  --threshold 0.5 \
  --seq_len 512 \
  --batch_size $BS \
  --epochs 0 \
  --test_log_file "$3" \
  --save_predictions "output/$4/$5/$4.$SLURM_ARRAY_TASK_ID"

seff $SLURM_JOBID
echo "END: $(date)"
