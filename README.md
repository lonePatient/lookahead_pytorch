## Lookahead Pytorch

pytorch implement of [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

## usage

```python
from optimizer import Lookahead
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer=base_optimizer,k=5,alpha=0.5)
```
## example

Data: CIFAR10

Model: ResNet_18

Epochs: 30

Optimizer: Adam

1. Run `python run.py --optimizer=adam` .
2. Run `python run.py --optimizer=lookahead` .

## result

### train loss

![](./png/loss.png)

## valid loss

![](./png/valid_loss.png)

## valid acc

![](./png/valid_acc.png)
