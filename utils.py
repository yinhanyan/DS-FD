def time_based_sliding_window_agg(arr, timestamps, N, func, ind):
    i, j = 0, 1
    sum = arr[0]
    l = len(arr)
    ret = ind
    while j < l:
        if timestamps[j-1] - timestamps[i] < N:
            sum += arr[j]
            j += 1
        elif timestamps[j-1] - timestamps[i] > N:
            sum -= arr[i]
            i += 1
        else:
            ret = func(ret, sum)
            sum += arr[j]
            j += 1

    return ret


def sliding_window_agg(arr, N, func, ind):
    i, j = 0, 1
    sum = arr[0]
    l = len(arr)
    ret = ind
    while j < l:
        if j-1-i < N:
            sum += arr[j]
            j += 1
        elif j-1-i > N:
            sum -= arr[i]
            i += 1
        else:
            ret = func(ret, sum)
            sum += arr[j]
            j += 1

    return ret
