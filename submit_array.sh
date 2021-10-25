#!/bin/bash
num=1
while true;
    do 
    count=$(squeue -u oinonenm | egrep -c 'predict')
    
    if [[ $count -gt 0 ]]
    then
        echo 'Not submitting, ' $count;
        date
        sleep 5m
    else
        if [[ $num -lt 14 ]]
        then
            echo 'submitting:'
            echo $num
            sbatch predict_array.sh en models/xlmrL.h5 test.log $1 $num 
            ((num=num+1))
            date
        else
            echo 'exiting';
            date
            exit 1
        fi
    fi
done
