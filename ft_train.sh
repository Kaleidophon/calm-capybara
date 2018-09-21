#!/usr/bin/env bash
# all rights reserved.
#
# this source code is licensed under the bsd-style license found in the
# license file in the root directory of this source tree. an additional grant
# of patent rights can be found in the patents file in the same directory.

set -e

DATADIR=data/
FASTTEXTDIR=fastText/

# compile
pushd $FASTTEXTDIR
make opt
popd 
ft=${FASTTEXTDIR}/fasttext

g++ -std=c++0x eval.cpp -o eval

# Train model and evaluate on the test set
dim=100 
epoch=100 
neg=100
model=data/us
pred=data/us_pred

echo "---- train ----"
$ft supervised -input $DIR/ft_us_train.text -dim $dim -epoch $epoch -output ${model} -lr .2 -thread 20 -loss ns -neg $neg -minCount 0



