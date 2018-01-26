import numpy as np
from collections import defaultdict
import sys

max_value = 8192
num_of_perm = 1020
num_of_bands = 34
prime = 1726943
prime2 = 923591
rows_per_band = num_of_perm / num_of_bands

np.random.seed(7777)
a_prim = np.random.randint(1, max_value, num_of_perm)
b_prim = np.random.randint(0, max_value, num_of_perm)


def mapper(key, value):
    tokens = value.split()
    page_str = tokens[0]
    shingles = tokens[1:]
    page_id = int(page_str[len('PAGE_'):])
    signature_mat_col = num_of_perm*[sys.maxint]
    for sh in shingles:
        tmps = np.mod(np.mod(np.add(np.multiply(a_prim, int(sh)), b_prim), prime), max_value)
        signature_mat_col = np.minimum(signature_mat_col, tmps)
        #for i in range(n_hash_funs):
        #    tmp = ((a_prim[i] * int(sh) + b_prim[i]) % prime) % n_rows
        #    signature_mat_col[i] = min(signature_mat_col[i], tmp)

    # split the signature_mat_col into bands and make band_id a key and value [page_ID, [signatures for that band]]
    r = num_of_perm / num_of_bands
    for b in range(num_of_bands):
        band = signature_mat_col[b*r:(b+1)*r]
        yield b, [page_id, band]

def generate_hash(band):
    return hash(str(band))


def reducer(key, values):
    # key: key from mapper used to aggregate
    bucket_to_page = defaultdict(list)
    for doc in values:
        page_id = doc[0]
        band = doc[1:]
        band_hash = generate_hash(band)
        bucket_to_page[band_hash].append(page_id)

    for bucket, page_id in bucket_to_page.iteritems():
        page_id.sort()
        for i in range(len(page_id) - 1):
            for j in range(i + 1, len(page_id)):
                yield page_id[i], page_id[j]


















































