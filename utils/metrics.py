from keras_nlp.metrics import Perplexity
from keras.metrics import SparseCategoricalAccuracy

def get_metrics(mask_token_id=0):
    """
    Returns standard evaluation metrics for language modeling.

    Args:
        mask_token_id (int): Token ID to ignore during perplexity calculation.

    Returns:
        list: List of compiled metrics.
    """
    return [
        Perplexity(from_logits=True, mask_token_id=mask_token_id),
        SparseCategoricalAccuracy(name="accuracy"),
    ]
