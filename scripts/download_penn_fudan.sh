#!/usr/bin/env bash
set -e
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip -q PennFudanPed.zip
rm -f PennFudanPed.zip
echo "Downloaded PennFudanPed/"
