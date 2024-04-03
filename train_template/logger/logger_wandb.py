import wandb
from dotenv import load_dotenv
import os

""" logging with wandb """
load_dotenv()

def train_logging(dev_loss: float, 
                  train_loss: float):
    
  # logging dev/train loss
  wandb.log({"train loss/epoch": train_loss, "dev loss/epoch": dev_loss})  