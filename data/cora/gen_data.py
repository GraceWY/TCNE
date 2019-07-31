import os
import sys

edge_fn='edge.txt'
o_entity_fn='entity.txt'
tag_fn='tag-edge.txt'
o_tag_fn='tag.txt'

def main():
    max_no = map_entity(edge_fn)
    write_mp_file(max_no, o_entity_fn)
    max_no = map_tag(tag_fn)
    write_mp_file(max_no, o_tag_fn)


def map_entity(edge_fn):
    ens = set()
    with open(edge_fn, "r") as fn:
        for line in fn:
            line = line.strip()
            items = line.split()
            ens.add(int(items[0]))
            ens.add(int(items[1]))

    return max(ens)


def map_tag(tag_fn):
    tags = set()
    with open(tag_fn, "r") as fn:
        for line in fn:
            line = line.strip()
            items = line.split()
            tags.add(int(items[1]))

    return max(tags)


def write_mp_file(max_no, fn):
    with open(fn, 'w') as f:
        for i in range(max_no+1):
            f.write(str(i) + "\t" + str(i) + "\n")


if __name__=="__main__":
    main()
