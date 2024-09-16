
for num_class in 1 8 16 32; do
    for seed in 42 43 44 45 46; do
        echo "Running with num_class=$num_class and seed=$seed"
        python qwn2_7b_train.py --seed $seed --num_class $num_class
    done
done
