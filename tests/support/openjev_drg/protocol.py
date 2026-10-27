import hashlib
import json
from itertools import product
import numpy as np

ROUTES = np.asarray([(0, *tail) for tail in product(range(3), repeat=3)], dtype=np.int64)

def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def make_members(frame, size):
    batches = []
    for b, start in enumerate(range(0, len(frame), size)):
        ids = frame.iloc[start:start+size].sample_id.astype(str).tolist()
        batches.append({'batch_id': b, 'start': start, 'stop': start+len(ids),
                        'sample_ids': ids, 'n': len(ids), 'sha256': content_hash(ids)})
    return {'logical_batch_size': size, 'num_examples': len(frame), 'batches': batches,
            'sha256': content_hash(batches)}

def conditional_oracle(costs, prefix):
    costs = np.asarray(costs, dtype=np.float64)
    prefix = tuple(int(x) for x in prefix)
    if costs.shape != (27,) or not np.isfinite(costs).all() or len(prefix) not in (1,2,3) or prefix[0] != 0:
        raise ValueError('Expected finite 27-route costs and an active-block prefix')
    q = np.asarray([costs[np.all(ROUTES[:, :len(prefix)+1] == (*prefix, a), axis=1)].min() for a in range(3)])
    return int(q.argmin()), q, q-q.min()
