# train
nohup python -u main.py --mode 'train' --num_epochs 100 --batch_size 512 \
               --dropout 0.1 --num_layers 3 --save_path './ckpt_l3/' > train.log 2>&1 &

# train ddp
#nohup python -u main.py --mode 'train_ddp' --num_epochs 100 --batch_size 512 \
#               --dropout 0.1 --num_layers 3 --save_path './ckpt_ddp/' \
#               --world_size 4 > train_ddp.log 2>&1 &
