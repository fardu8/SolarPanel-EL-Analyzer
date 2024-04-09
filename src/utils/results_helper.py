def map_label_to_probability(label):
    if label == 0:
        return 'fully functional (0% probability of being defective)'
    elif label == 1:
        return ' possibly defective (33% probability)'
    elif label == 2:
        return 'likely defective (67% probability)'
    elif label == 3:
        return 'certainly defective (100% probability)'
    else:
        raise ValueError('Invalid label')
