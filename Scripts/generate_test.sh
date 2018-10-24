#!/bin/bash

fileName=test_todo.sh
epochs=50
epoch_Review=50
epoch_G=1
num_task=10


rm $fileName
echo '#!/bin/bash' >> $fileName
chmod +x $fileName

datasets="mnist fashion"
models="WGAN VAE CVAE CGAN GAN WGAN_GP" #BEGAN
methods="Baseline Generative_Replay Reharsal Ewc" #
seeds="0 1 2 3 4 5 6 7"
seeds="0"


########################## GENERATE INPUT DATA FILES ###########################

echo 'cd ../Data' >> $fileName

dir="../Archives"

echo '#Generate Inpu Data' >> $fileName
for dataset in $datasets ;do
echo '#'$dataset >> $fileName
echo '#For the expert' >> $fileName
echo python main.py --task disjoint --dataset $dataset --n_tasks 1 --dir $dir >> $fileName

echo '#For the models to train' >> $fileName
echo python main.py --task disjoint --dataset $dataset --n_tasks $num_task  --dir $dir >> $fileName

echo '#For Upperbound' >> $fileName
echo python main.py --task disjoint --upperbound True --dataset $dataset --n_tasks $num_task  --dir $dir >> $fileName
done #datasets




############### EXPERTS   ######################

dir="./Archives"
echo 'cd ..' >> $fileName


for seed in $seeds; do


    #fileName=test_todo$seed\.sh

    #rm $fileName
    #echo '#!/bin/bash' >> $fileName
    #echo 'cd ..' >> $fileName  # Go back to main folder

    for dataset in $datasets ;do

        #if [ "$dataset" == "mnist" ]; then
        #    fileName=test_todo$seed\.sh
        #elif [ "$dataset" == "fashion" ]; then
        #    fileName=test_todo$seed\bis.sh
        #fi
        #rm $fileName
        #echo '#!/bin/bash' >> $fileName
        #echo 'cd ..' >> $fileName  # Go back to main folder
        #chmod +x $fileName



        echo python main.py --context Classification --task_type disjoint --method Baseline --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --num_task 1 --seed $seed  --dir $dir >> $fileName
        ###############  TEST TODO ######################

        echo '#'$dataset >> $fileName

        for model in $models ;do
            echo '#'$model >> $fileName

            for method in  $methods;do
            echo '#'$method >> $fileName

            echo python main.py --context Generation --task_type disjoint --method $method --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model  --train_G True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method $method --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --FID True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method $method --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --Fitting_capacity True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method $method --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --trainEval True --seed $seed  --dir $dir >> $fileName


            done #method


            ################  UPPERBOUND ####################


            echo '#Upperbound #' >> $fileName
            echo '#'$dataset >> $fileName

            echo python main.py --context Generation --task_type disjoint --method Baseline --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --upperbound True --train_G True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method Baseline --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --upperbound True --FID True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method Baseline --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --upperbound True --Fitting_capacity True --seed $seed  --dir $dir >> $fileName
            echo python main.py --context Generation --task_type disjoint --method Baseline --dataset $dataset --epochs $epochs --epoch_Review $epoch_Review --epoch_G $epoch_G --num_task $num_task --gan_type $model --upperbound True --trainEval True --seed $seed  --dir $dir >> $fileName

        done #model

    done #dataset

done #seed
