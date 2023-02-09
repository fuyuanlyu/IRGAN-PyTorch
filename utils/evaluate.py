import numpy as np

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def evaluate_one_user(x):
    rating = x[0]
    u = x[1]
    user_pos_train = x[2]
    user_pos_valid = x[3]
    all_items = x[4]
    topks = [3, 5, 10]
    topk = max(topks)

    cand_items = list(set(all_items) - set(user_pos_train))
    item_score = []
    for i in cand_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_valid:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:topks[0]])
    p_5 = np.mean(r[:topks[1]])
    p_10 = np.mean(r[:topks[2]])

    ndcg_3 = ndcg_at_k(r, topks[0])
    ndcg_5 = ndcg_at_k(r, topks[1])
    ndcg_10 = ndcg_at_k(r, topks[2])

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

