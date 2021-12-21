#!/bin/bash

DATA_URL=$(curl -X GET -H "Authorization: $1" -H "Accept: application/json" https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/5C2Ll9-_xhfshQ | awk -F \" '{print $4}')
wget "$DATA_URL" -O data.zip
unzip data.zip -d input
mv 'input/Итоговый проект'/* input
rm -fR 'input/Итоговый проект'
rm -f data.zip