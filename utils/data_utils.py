import random
import hashlib
import numpy as np

def get_unk_mask_indices(image, testing, num_labels, known_labels, epoch=1):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels - int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0, int(num_labels * 0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels - num_known))

    return unk_mask_indices