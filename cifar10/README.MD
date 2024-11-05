
## CIFAR 10/100 Training

- model: resnet20 / resnet32 / resnet110 / densenet / wrn

```
python main.py --model resnet20 --optim sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0005 --epoch 10 --run 0;
python main.py --model resnet20 --optim kfac --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 0.03 --tcov 5 --tinv 50 --epoch 10 --run 0;
python main.py --model resnet20 --optim eva --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 0.03 --tcov 5 --tinv 50 --epoch 10 --run 0;
python main.py --model resnet20 --optim foof --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --epoch 10 --run 0;
python main.py --model resnet20 --optim mac --lr 0.01 --momentum 0.9 --weight_decay 0.05 --damping 0.1 --tinv 100 --epoch 10 --run 0;
python main.py --model resnet20 --optim nysact_g --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --rank 10 --epoch 10 --run 0;
python main.py --model resnet20 --optim nysact_s --lr 0.1 --momentum 0.9 --stat_decay 0.95 --weight_decay 0.0005 --damping 1.0 --tcov 5 --tinv 50 --rank 10 --epoch 10 --run 0;
```
