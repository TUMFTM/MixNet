from random import randint


def quicksort(array, compare_fn):

    # If the input array contains fewer than two elements,
    # then return it as the result of the function

    if len(array) < 2:

        return array

    low, same, high = [], [], []

    # Select your `pivot` element randomly
    pivot = array[randint(0, len(array) - 1)]

    for item in array:
        # Elements that are smaller than the `pivot` go to
        # the `low` list. Elements that are larger than
        # `pivot` go to the `high` list. Elements that are
        # equal to `pivot` go to the `same` list.
        if item == pivot:
            same.append(item)
        elif compare_fn(item, pivot):
            low.append(item)
        else:
            high.append(item)

    # The final result combines the sorted `low` list
    # with the `same` list and the sorted `high` list

    return quicksort(low, compare_fn) + same + quicksort(high, compare_fn)


def sort_collisions(collision_list, priority_list):
    """Sort a collision list with a given priority list.

    Args:
        collision_list ([list]): [list of collision dicts]
        priority_list ([list]): [sorted list of prediction IDs according to their priority]

    Returns:
        [list]: [sorted list of collision dicts]
    """
    for collision in collision_list:
        collision["priority"] = priority_list.index(
            collision["pred_ids"][0]
        ) + priority_list.index(collision["pred_ids"][1])

        # Setting who is leading and who is following based on
        # the vehicle indices in the priority list:
        if priority_list.index(collision["pred_ids"][0]) > priority_list.index(
            collision["pred_ids"][1]
        ):
            collision["following_pred_id"] = collision["pred_ids"][0]
            collision["leading_pred_id"] = collision["pred_ids"][1]
        else:
            collision["following_pred_id"] = collision["pred_ids"][1]
            collision["leading_pred_id"] = collision["pred_ids"][0]

    collision_list_sorted = sorted(collision_list, key=lambda k: k["priority"])

    return collision_list_sorted
