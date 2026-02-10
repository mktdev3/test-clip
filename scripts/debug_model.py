
import torch
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "rinna/japanese-clip-vit-b-16"
logger.info(f"Loading model: {model_name}")
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "猫"
inputs = tokenizer(text, return_tensors="pt")

logger.info(f"Model type: {type(model)}")
logger.info(f"Inputs: {inputs.keys()}")

with torch.no_grad():
    try:
        features = model.get_text_features(**inputs)
        logger.info(f"Features type: {type(features)}")
        logger.info(f"Dir features: {dir(features)}")
        if hasattr(features, 'keys'):
            logger.info(f"Features keys: {features.keys()}")
    except Exception as e:
        logger.error(f"Error calling get_text_features: {e}")

    # Alternative
    try:
        outputs = model.text_model(**inputs)
        logger.info(f"Text model outputs type: {type(outputs)}")
        
        if hasattr(model, 'text_projection'):
            logger.info(f"text_projection type: {type(model.text_projection)}")
            # Try to print shape if it is a tensor or parameter
            if isinstance(model.text_projection, torch.Tensor):
                logger.info(f"text_projection shape: {model.text_projection.shape}")
            elif hasattr(model.text_projection, 'weight'):
                 logger.info(f"text_projection weight shape: {model.text_projection.weight.shape}")
        else:
            logger.info("No text_projection found on model")
            
    except Exception as e:
        logger.error(f"Error calling text_model: {e}")
