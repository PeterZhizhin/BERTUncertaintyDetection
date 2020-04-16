#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

train_output=$1
valid_output=$2
train_percentage=$3

all_files=$(printf "%s\n" ${@:4})
shuffled_files="$(echo "$all_files" | shuf --random-source=<(get_seeded_random 1337))"
number_of_files=$(echo "$shuffled_files" | wc -l)

number_of_train_files=$(expr $train_percentage \* $number_of_files / 100)
number_of_test_files=$(expr $number_of_files - $number_of_train_files)

echo "Number of train files: $number_of_train_files"
echo "Number of test files: $number_of_test_files"

train_files=$(echo "$shuffled_files" | head -n $number_of_train_files)
test_files=$(echo "$shuffled_files" | tail -n $number_of_test_files)

cat $train_files > $train_output
cat $test_files > $valid_output
