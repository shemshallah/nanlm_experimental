# v35 mechanism control

Local CPU PyTorch run: two binary signals; anchor chooses which signal is the target. 2048 training examples, 2048 separate generated validation examples, identical 240 minibatches of 64, identical parameter shapes and initial values, Adam lr .02. Uniform allocation vs selective allocation and routing. Only first two readout coordinates used as binary logits. This constructed task is not a language model or corpus result.

| Operator | Held-out accuracy | NLL (nats) |
| --- | ---: | ---: |
| Uniform | .7509765625 | .614194929599762 |
| Selective | .7509765625 | .5252373218536377 |

Probability calibration improved; classification accuracy did not. The toy task is deliberately sparse and does not establish benefits on NaNROUTER. The integrated v35 workflow 45903532512 is pending, dependent on v34. An independently trained matched-budget corpus control remains necessary.
