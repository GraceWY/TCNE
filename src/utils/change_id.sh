#!/bin/sh
DATA_PATH="/Users/wangyun/repos/TCNE/data/bilibili2/"
IN_PATH=$DATA_PATH"tag-edge.txt.ori"
OUT_PATH=$DATA_PATH"tag-edge.txt"

awk 'BEGIN {
    oid = "200";
    nid = "150";
}
{
    if ($2 == oid) {
        print $1"\t"nid
    }
    else {
        print $0
    }
}' $IN_PATH > $OUT_PATH  
