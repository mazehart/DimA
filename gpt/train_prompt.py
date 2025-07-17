from prompt import train_prompt
import argparse

parser = argparse.ArgumentParser()
task = {'xsum': {'epoch': 1, 'rec_step': 150},
        }
seed = 0
bsz = 16
plm = [
    # "gpt2", 
    #    "gpt2-medium", 
       "gpt2-large"
       ]
lr = {"gpt2":3e-3, "gpt2-medium":4e-4,"gpt2-large": 2e-4}

for each in task:
    for model in plm:
        train_prompt(task=each,
                     rec_step=task[each]['rec_step'],
                     seed=seed,
                     bsz=bsz,
                     lr=lr[model],
                     epoch=task[each]['epoch'],
                     plm=model,
                     tag=6,
                     )
