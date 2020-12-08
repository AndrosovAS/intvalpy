from __future__ import annotations

import json
import random
from functools import reduce, wraps
from itertools import product
from statistics import median, mode
from collections import Counter
from typing import List, Any, Iterable, Callable, Union, Dict


def _collect(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)
        return self.make(method_output)

    return _impl


class Collection:
    def __init__(self, items) -> None:
        self.og_item = items
        item_type = type(items)
        if item_type == list or item_type == dict:
            self.contents = items
        else:
            self.contents = [items]

    @_collect
    def all(self):
        return self.contents

    def append(self, item):
        return self.push(item)

    def avg(self, key=None):
        if key:
            return self.make(self.contents).pluck(key).sum() / len(self.contents)
        return self.make(self.contents).sum() / len(self.contents)

    @_collect
    def chunk(self, n):
        new_array = []
        for i in range(0, len(self.contents), n):
            new_array.append(self.contents[i:i + n])
        return new_array

    @_collect
    def combine(self, other):
        return self.make(self.contents) + self.make(other)

    def contains(self, item):
        if callable(item):
            return self.make(self.contents).first(item) in self.contents
        return item in self.contents

    def count(self) -> Collection:
        return len(self.contents)

    def count_by(self, *func):
        if func:
            items = self.make(self.contents).map(func[0])
        else:
            items = self.contents
        return Counter(items)

    def dd(self):
        exit(self.make(self.contents))

    def diff(self, lst):
        return self.make(self.contents).filter(lambda x: x not in lst)

    @_collect
    def duplicates(self):
        return self.make(list(Counter(self.contents).items())
                         ).filter(lambda x: x[1] > 1) \
            .map(lambda x: x[0])

    @_collect
    def each(self, function: Callable) -> Collection:
        for item in self.contents:
            function(item)
        return self.contents

    def every(self, func):
        if not self.contents:
            return True
        return self.make(self.contents).map(func).contains(False) == False

    @_collect
    def filter(self, function: Callable) -> List | Dict:
        try:
            return {
                k: v for k, v in list(
                    filter(
                        lambda x: function(*x), self.contents.items()
                    )
                )
            }
        except AttributeError:
            return list(filter(function, self.contents))

    def first(self, *func):
        if not self.contents:
            return None
        elif not func:
            return self.contents[0]

        for item in self.contents:
            if func[0](item):
                return item
        return False

    def first_where(self, key, value):
        return self.make(self.contents).first(lambda x: x[key] == value)

    @_collect
    def flatten(self, depth=None) -> Collection:
        new_items = []

        def append_item(item, depth):
            new_item = self.make(item) if type(item) != Collection else item
            for i in new_item:
                if type(i) != Collection and type(i) != list:
                    new_items.append(i)
                else:
                    depth = depth - 1 if depth is not None else depth
                    if depth is None or depth != 0:
                        append_item(i, depth)
                    else:
                        new_items.append(i)

        self.make(self.contents).each(lambda x: append_item(x, depth))
        return new_items

    def flat_map(self, func):
        return self.flatten().map(func)

    def for_page(self, page, size):
        return self.make(self.contents).chunk(size)[page - 1]

    def implode(self, glue):
        return glue.join(self.contents)

    def intersect(self, other):
        return self.make(self.contents).filter(lambda x: x in other)

    def is_empty(self):
        return len(self.contents) == 0

    def is_not_empty(self):
        return len(self.contents) > 0

    def key_by(self, key):
        return self.make(self.contents).map(lambda x: {key: x})

    def last(self, *func):
        if not self.contents:
            return None
        elif not func:
            return self.contents[-1]

        for item in reversed(self.contents):
            if func[0](item):
                return item
        return False

    @_collect
    def map(self, function: Callable) -> List | Dict:
        try:
            return {
                k: v for k, v in list(
                    map(
                        lambda x: function(*x), self.contents.items()
                    )
                )
            }
        except AttributeError:
            return list(map(function, self.contents))

    def map_into(self, cls):
        return self.make(self.contents).map(lambda x: cls(x))

    def map_spread(self, func):
        return self.make(self.contents).map(lambda args: func(*args)).flatten()

    def max(self, field=None):
        return self.pluck_and_func(max, field)

    def median(self, field=None):
        return self.pluck_and_func(median, field)

    def mode(self, field=None):
        return self.pluck_and_func(mode, field)

    @_collect
    def merge(self, other):
        try:
            return self + other
        except AttributeError:
            return self.contents + other

    def min(self, field=None):
        return self.pluck_and_func(min, field)

    def nth(self, place, offset=0):
        return self.make(self.contents) \
            .skip(offset) \
            .enumerate() \
            .filter(lambda x: x[0] % place == 0) \
            .map(lambda x: x[1])

    def pad(self, size, left=False, pad_char=0):
        padded = []
        for _ in range(size - self.make(self.contents).count()):
            if left:
                padded.append(pad_char)
            else:
                self.contents.append(pad_char)
        if left:
            padded.append(self.contents)
            return self.make(padded).flatten()
        else:
            return self.make(self.contents)

    def partition(self, func):
        collection = self.make(self.contents)
        true_condition = collection.filter(func)
        false_condition = collection.reject(func)
        return true_condition, false_condition

    def pop(self):
        last_item = self.contents.pop()
        return last_item

    def prepend(self, item):
        self.contents.insert(0, item)
        return self

    def push(self, item):
        self.contents.append(item)
        return self

    @_collect
    def pipe(self, func):
        return func(self.contents)

    @_collect
    def pluck(self, attr: str) -> List:
        return list(map(lambda x: x[attr], self.contents))

    def random(self, retrieve=1):
        return random.choices(self.contents, k=retrieve) if retrieve != 1 else random.choice(self.contents)

    @_collect
    def reduce(self, function: Callable, accumulator: Any) -> List:
        return list(reduce(function, self.contents, accumulator))

    def reject(self, func):
        return self.make(self.contents).filter(lambda x: not func(x))

    @_collect
    def reverse(self):
        return list(reversed(self.contents))

    @_collect
    def enumerate(self):
        return list(enumerate(self.contents))

    def search(self, search):
        return self.make(self.contents) \
            .enumerate() \
            .first(lambda x: search == x[1])[0]

    def shift(self):
        self.contents = self.contents[1:]
        return self

    @_collect
    def shuffle(self):
        return random.sample(self.contents, k=len(self.contents))

    @_collect
    def skip(self, offset):
        return self.contents[offset:]

    def skip_until(self, func):
        return self.__skip_base(func, 0)

    def skip_while(self, func):
        return self.__skip_base(func, 1)

    def __skip_base(self, func, places):
        collection = self.make(self.contents)
        if callable(func):
            place = collection.enumerate().first(lambda x: func(x[1]))[0]
        else:
            place = collection.enumerate().first(lambda x: x[1] == func)[0]
        return collection.skip(place + places)

    def slice(self, position, size=None):
        if not size:
            return self.skip(position)
        else:
            return self.make(self.contents[position:position + size])

    def sum(self, field: str = None) -> Collection:
        return self.pluck_and_func(sum, field)

    @_collect
    def sort(self, func=None):
        return sorted(self.contents, key=func)

    @_collect
    def sort_desc(self, func=None):
        return sorted(self.contents, key=func, reverse=True)

    @_collect
    def sort_by(self, key):
        return sorted(self.contents, key=lambda x: x[key])

    @_collect
    def sort_by_desc(self, key):
        return sorted(self.contents, key=lambda x: x[key], reverse=True)

    @_collect
    def splice(self, pos, size=0, *args):
        to_return = self.contents[pos:pos + size]
        temp_list = []
        for key, item in enumerate(self.contents):
            if args and pos <= key < (pos + size):
                self.contents[key] = args[0]
            if not pos <= key < (pos + size):
                temp_list.append(item)
        if not args:
            self.contents = temp_list
        return to_return

    @_collect
    def split(self, size):
        lists = []
        d, r = divmod(len(self.contents), size)
        for i in range(size):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            lists.append(self.contents[si:si + (d + 1 if i < r else d)])
        return lists

    @_collect
    def take(self, count):
        if count > 0:
            return self.contents[:count]
        else:
            return self.contents[count:]

    @_collect
    def take_until(self, func):
        collection = self.make(self.contents)
        if callable(func):
            place = collection.enumerate().first(lambda x: func(x[1]))[0]
        else:
            place = collection.enumerate().first(lambda x: x[1] == func)[0]
        return self.contents[:place]

    @_collect
    def take_while(self, func):
        first = self.make(self.contents).first(lambda x: not func(x))
        return self.contents[:first - 1]

    def tap(self, func):
        func(self.make(self.contents))
        return self

    def pluck_and_func(self, func, field=None):
        if field:
            return func(self.make(self.contents).pluck(field).to_list())
        else:
            return func(self.contents)

    @classmethod
    def times(cls, times, func):
        new_list = []
        for number in range(1, times + 1):
            new_list.append(func(number))
        return cls(new_list)

    def to_json(self):
        return json.dumps(self.contents)

    def transform(self, func):
        self.contents = list(map(func, self.contents))
        return self

    @_collect
    def unique(self):
        return list(set(self.contents))

    def unless(self, condition, func):
        if not condition:
            return func(self.make(self.contents))
        else:
            return self

    def unless_empty(self, func):
        return self.when_not_empty(func)

    def unless_not_empty(self, func):
        return self.when_empty(func)

    @staticmethod
    def unwrap(item):
        try:
            return item.og_item
        except AttributeError:
            return item

    def when(self, condition, func):
        if condition:
            return func(self.make(self.contents))
        else:
            return self

    def when_not_empty(self, func):
        if len(self.contents) != 0:
            return func(self.make(self.contents))
        else:
            return self

    def when_empty(self, func):
        if len(self.contents) == 0:
            return func(self.make(self.contents))
        else:
            return self

    def where(self, key, value):
        return self.make(self.contents).filter(lambda x: x[key] == value)

    def where_between(self, key, range_list):
        return self.make(self.contents).filter(lambda x: range_list[0] <= x[key] <= range_list[1])

    def where_in(self, key, options):
        return self.make(self.contents).filter(lambda x: x[key] in options)

    def where_instance_of(self, cls):
        return self.make(self.contents).filter(lambda x: isinstance(x, cls))

    def where_not_between(self, key, range_list):
        return self.make(self.contents).filter(lambda x: not range_list[0] <= x[key] <= range_list[1])

    def where_not_in(self, key, options):
        return self.make(self.contents).filter(lambda x: x[key] not in options)

    def where_not_null(self):
        return self.make(self.contents).filter(lambda x: x is not None)

    def where_null(self):
        return self.make(self.contents).filter(lambda x: x is None)

    @classmethod
    def wrap(cls, item):
        if isinstance(item, cls):
            return item
        else:
            return cls(item)

    @_collect
    def zip(self, other):
        return list(zip(self.contents, other))

    def to_list(self) -> List:
        return self.contents

    @classmethod
    def make(cls, items) -> Collection:
        if isinstance(items, cls):
            return items
        else:
            return cls(items)

    @_collect
    def keys(self):
        return list(self.contents.keys())

    def diff_assoc(self, other):
        return self.make(self.contents) \
            .filter(lambda x, y: (x, y) not in other.items())

    def diff_keys(self, other):
        return self.make(self.contents) \
            .filter(lambda x, y: x not in other.keys())

    @_collect
    def except_for(self, except_keys):
        return {
            k: v for k, v in self.contents.items() if k not in except_keys
        }

    @_collect
    def flip(self):
        return {v: k for k, v in self.contents.items()}

    def forget(self, key):
        self.contents = {k: v for k,
                         v in self.contents.items() if k is not key}
        return self

    def get(self, key):
        return self.contents[key]

    def has(self, key):
        return key in self.contents.keys()

    @_collect
    def map_to_groups(self, func):
        grouped = {}
        for item in self.contents:
            key, value = list(func(item).items())[0]
            if key not in grouped.keys():
                grouped[key] = []
            grouped[key].append(value)
        return grouped

    @_collect
    def map_with_keys(self, func):
        temp = {}
        for item in self.contents:
            k, v = list(func(item).items())[0]
            temp[k] = v
        return temp

    @_collect
    def only(self, keys):
        return {k: v for k, v in self.contents.items() if k in keys}

    def pull(self, key):
        to_return = self.contents[key]
        del self.contents[key]
        return to_return

    def put(self, key, value):
        self.contents[key] = value
        return self

    @_collect
    def replace(self, to_replace):
        return self.__replace_base(self.contents, to_replace)

    @_collect
    def replace_recursive(self, to_replace):
        return self.__replace_base(self.contents, to_replace, recursive=True)

    def __replace_base(self, original, to_replace, recursive=False):
        contents = {k: v for k, v in enumerate(original)}
        for k, v in to_replace.items():
            if type(v) == dict and recursive == True:
                contents[k] = self.__replace_base(contents[k], to_replace[k])
            else:
                contents[k] = v
        return list(contents.values())

    @_collect
    def sort_by_keys(self):
        return {k: self.contents[k] for k in sorted(self.contents)}

    @_collect
    def sort_by_keys_desc(self):
        return {k: self.contents[k] for k in sorted(self.contents, reverse=True)}

    @_collect
    def items(self):
        return list(self.contents.items())

    @_collect
    def values(self):
        return list(self.contents.values())

    @_collect
    def group_by(self, key):
        grouped = {}
        for item in self.contents:
            grouped_by = item[key]
            if key not in grouped.keys():
                grouped[grouped_by] = [item]
            else:
                grouped[grouped_by].append(item)
        return grouped

    @_collect
    def cross_join(self, *others):
        return [list(x) for x in list(product(self.contents, *others))]

    def each_spread(self, func):
        for item in self.contents:
            result = func(*item)
            if result is False:
                break

    @_collect
    def intersect_by_keys(self, other):
        return {k: v for k, v in self.contents.items() if k in other.keys()}

    def __eq__(self, other):
        return self.contents == other.contents

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.to_list()})'

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return self.count()

    def __getitem__(self, key) -> Any:
        return self.contents[key]

    def __setitem__(self, key, value) -> None:
        self.contents[key] = value

    def __iter__(self) -> Iterable:
        return self.contents.__iter__()

    def __add__(self, other):
        return self.contents + other.contents
