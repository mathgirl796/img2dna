import collections


def count_sort(arr):
    return sorted(collections.Counter(arr).items(), key=lambda x: x[0])