##########################################
################ _wandb.py ###############
##########################################
import os
import json
import sys
import wandb

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

def build_wandb(wandb_json_path, kaggle_env, dir, project, name, config, group):
    try:
        if kaggle_env:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            secret_value_0 = user_secrets.get_secret("wandb_api")
        else:
            f = open(wandb_json_path, "r")
            json_data = json.load(f)
            secret_value_0 = json_data["wandb_api"]
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
    

    run = wandb.init(dir=dir,
                     project=project, 
                     name=name,
                     config=class2dict(config),
                     group=group,
                     job_type="train",
                     anonymous=anony)
    return run