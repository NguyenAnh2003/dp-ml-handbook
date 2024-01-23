import wandb
from dotenv import load_dotenv
import os

""" logging with wandb """
load_dotenv()

wandb.login(key=os.getenv("WANDB_API"))