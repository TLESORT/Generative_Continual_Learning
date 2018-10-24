#!/bin/bash
cd ..

python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 0 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 0 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 0 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 0 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 0 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 0 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 0 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 0 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 0 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 0 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 1 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 1 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method regenerate --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 1 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 1 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 1 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 1 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 1 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 1 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 1 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 1 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 2 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 2 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 2 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 2 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 2 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 2 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 2 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 2 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 2 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 2 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 3 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 3 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 3 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 3 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 3 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 3 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 3 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 3 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 3 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 3 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 4 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 4 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 4 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 4 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 4 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 4 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 4 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 4 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 4 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 4 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 5 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 5 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 5 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 5 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 5 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 5 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 5 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 5 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 5 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 5 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 6 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 6 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 6 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 6 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 6 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 6 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 6 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 6 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 6 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 6 --dir ./Archives
python main.py --context Classification --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --num_task 1 --seed 7 --dir ./Archives
#mnist
#WGAN_GP
#Baseline
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 7 --dir ./Archives
#Generative_Transfer
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Generative_Transfer --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 7 --dir ./Archives
#Reharsal_balanced
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Reharsal_balanced --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 7 --dir ./Archives
#Upperbound #
#mnist
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --train_G True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --FID True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --Fitting_capacity True --seed 7 --dir ./Archives
python main.py --context Generation --task_type disjoint --method Baseline --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --upperbound True --trainEval True --seed 7 --dir ./Archives
# EWC_EWC SAMPLES #
#5
#WGAN_GP
#Ewc
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --train_G True --seed 7 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --FID True --seed 7 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --Fitting_capacity True --seed 7 --dir ./Archives
python main.py --lambda_EWC 5 --context Generation --task_type disjoint --method Ewc --dataset mnist --epochs 50 --epoch_Review 50 --epoch_G 1 --num_task 10 --gan_type WGAN_GP --trainEval True --seed 7 --dir ./Archives
