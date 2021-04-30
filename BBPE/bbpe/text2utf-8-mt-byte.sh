set -e

TEXT_DIR=./MTData/
NUM=0
for TEXT_FILE in ${TEXT_DIR}/*; do
NUM=$((NUM+1))
echo $NUM
cat $TEXT_FILE | python3 utf-8-mt-byte.py MTData_byte/$(basename "$TEXT_FILE") &
if (($NUM>60))
then
wait
NUM=0
fi
done

