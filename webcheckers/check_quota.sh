#!/bin/sh

curl -s http://www.cic.gc.ca/english/work/iec/data.xml | grep "$1.*$2" -A3 | tail -n 2 | grep -o "[0-9]*"
