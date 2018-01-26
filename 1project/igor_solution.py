import sys
import numpy as np
from collections import defaultdict

prime = 1726943
prime2 = 923591
n_hash_funs = 1012
n_rows = 8192
n_bands = 44
n_buckets = 17269430

np.random.seed(7777)
a_prim = np.random.randint(1, n_rows, n_hash_funs)
b_prim = np.random.randint(0, n_rows, n_hash_funs)



"""
    input:
        key: None
        value: line of the input file
    output:
        key: band id
        value: [page_id, [signatures in band i], [page shingles]]]

"""
def mapper(key, value):
    tokens = value.split()
    page_str = tokens[0]
    shingles = map(int, tokens[1:])
    page_id = int(page_str[len('PAGE_'):])

    signature_mat_col = n_hash_funs*[sys.maxint]
    for sh in shingles:
        tmps = np.mod(np.mod(np.add(np.multiply(a_prim, sh), b_prim), prime), n_rows)
        signature_mat_col = np.minimum(signature_mat_col, tmps)

    # split the signature_mat_col into bands and make band_id a key
    # and value [page_ID, [signatures for that band], [page shingles]]
    r = n_hash_funs / n_bands
    for b in range(n_bands):
        band = signature_mat_col[b*r:(b+1)*r]
        yield b, [page_id, band, shingles]

"""
    input:
        key: band id
        values: list of form [[page_id, [signatures in band i], [page shingles]]]
    output:
        key: lower PAGE_ID
        value: higher PAGE_ID
"""
def reducer(key, values):
    r = n_hash_funs / n_bands
    np.random.seed(key)
    a = np.random.randint(1, n_rows, r)
    b = np.random.randint(0, n_rows, r)
    bucket_to_page = defaultdict(list)
    page_to_sh = {}
    for doc in values:
        page_id = doc[0]
        column = doc[1]
        page_to_sh[page_id] = doc[2]
        band_hash = sum([(a[i]*column[i] +b[i] % prime2) % n_buckets for i in range(r)]) % n_buckets
        bucket_to_page[band_hash].append(page_id)

    for bucket, page_id in bucket_to_page.iteritems():
        page_id.sort()
        for i in range(len(page_id)-1):
            for j in range(i+1, len(page_id)):
                col1 = page_to_sh[page_id[i]]
                col2 = page_to_sh[page_id[j]]

                a_and_b = np.shape(np.intersect1d(col1, col2))[0]
                a_or_b = np.shape(np.union1d(col1, col2))[0]
                if a_and_b > a_or_b:
                    print 'Warn: there might be wrong jaccard meassured!'

                dist = 0.0
                if a_and_b > 0.0:
                    dist = float(a_and_b) / float(a_or_b)
                if dist >= 0.85:
                    yield page_id[i], page_id[j]