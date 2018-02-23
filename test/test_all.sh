#!/bin/sh
for net_name in simple_net table_net upsampling_net
do
  bash test/test_net.sh $net_name > /dev/null
  if [ ! $? -eq 0 ]; then
      echo $net_name: failed
  else
      echo $net_name: passed
  fi
done
