python -u main.py --model A2A --data ETTh1 --features S --label_len 168 --pred_list 96 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.05 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data ETTh1 --features S --label_len 168 --pred_list 192,336,720 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data ETTh1 --features M --label_len 168 --pred_list 96 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.05 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5 --instance_loss

python -u main.py --model A2A --data ETTh1 --features M --label_len 168 --pred_list 192,336,720 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5 --instance_loss

python -u main.py --model A2A --data ETTm2 --features S --label_len 168 --pred_list 96 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.05 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data ETTm2 --features S --label_len 168 --pred_list 192,336,720 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data ETTm2 --features M --label_len 168 --pred_list 96 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.05 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5 --instance_loss

python -u main.py --model A2A --data ETTm2 --features M --label_len 168 --pred_list 192,336,720 --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --cost_epochs 3 --cost_grow_epochs 0 --itr 5 --instance_loss

python -u main.py --model A2A --root_path ./data/ECL/ --data ECL --features S --label_len 672 --pred_list 96,192,336,720 --d_model 60 --repr_dim 640 --kernel 3 --attn_nums 4 --pyramid 4 --criterion Standard --dropout 0.05 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Tanh --target MT_321 --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --root_path ./data/ECL/ --data ECL --features M --label_len 672 --pred_list 96,192,336,720 --d_model 60 --repr_dim 640 --kernel 3 --attn_nums 4 --pyramid 4 --criterion Standard --dropout 0.05 --batch_size 1 --cost_batch_size 1 --order_num 1 --aug_num 3 --jitter 0.1 --activation Tanh --target MT_321 --cost_epochs 3 --cost_grow_epochs 0 --instance_loss --itr 5

python -u main.py --model A2A --root_path ./data/Traffic/ --data Traffic --features S --label_len 672 --pred_list 96,192,336,720 --d_model 60 --repr_dim 640 --kernel 3 --attn_nums 4 --pyramid 4 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --target Sensor_861 --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --root_path ./data/Traffic/ --data Traffic --features M --label_len 672 --pred_list 96,192,336,720 --d_model 60 --repr_dim 640 --kernel 3 --attn_nums 4 --pyramid 4 --criterion Standard --dropout 0.05 --batch_size 1 --cost_batch_size 1 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu --target Sensor_861 --cost_epochs 3 --cost_grow_epochs 0 --instance_loss --itr 5

python -u main.py --model A2A --data Exchange --root_path ./data/Exchange/ --features S --label_len 30 --pred_list 96,192,336,720  --d_model 32 --repr_dim 640 --kernel 3 --attn_nums 1 --pyramid 1 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu  --target Singapore --cost_epochs 1 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data Exchange --root_path ./data/Exchange/ --features M --label_len 30 --pred_list 96,192,336,720  --d_model 32 --repr_dim 640 --kernel 3 --attn_nums 1 --pyramid 1 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Gelu  --target Singapore --cost_epochs 1 --cost_grow_epochs 0 --itr 5 --instance_loss

python -u main.py --model A2A --data weather --root_path ./data/weather/ --features S --label_len 96 --pred_list 96,192,336,720  --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Tanh --cost_epochs 3 --cost_grow_epochs 0 --itr 5

python -u main.py --model A2A --data weather --root_path ./data/weather/ --features M --label_len 96 --pred_list 96,192,336,720  --d_model 48 --repr_dim 640 --kernel 3 --attn_nums 3 --pyramid 3 --criterion Standard --dropout 0.1 --batch_size 32  --cost_batch_size 32 --order_num 1 --aug_num 3 --jitter 0.1 --activation Tanh --cost_epochs 3 --cost_grow_epochs 0 --itr 5 --instance_loss
