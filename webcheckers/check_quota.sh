#!/bin/sh

quotas=$(curl -s http://www.cic.gc.ca/english/work/iec/data.xml | grep "$1.*$2" -A3 | tail -n 2 | grep -o "[0-9]*")

for q in $quotas
do
	if [ "$q" != 0 ]
	then
		echo "Quotas are now open"
		echo "$quotas"
		break;
	fi
done

if [ -z "$quotas" ]
then
	echo "Quota check failed"
fi


