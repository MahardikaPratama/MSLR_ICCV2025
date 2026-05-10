#!/bin/bash

if [ -z "$2" ];then
echo "preprocess_bisindo.sh <hypothesis-CTM-file> <tmp-cmt-file> <output-file>"
exit 0
fi

hypothesisCTM=$1
tmpFile=$2
output=$3

# For BISINDO: minimal preprocessing - only remove empty markers and special tokens
echo "preprocess_bisindo.sh ${hypothesisCTM} ${tmpFile} ${output}"
cat ${hypothesisCTM} | grep -v "__EMOTION__" | grep -v "__EPENTHESIS__" | grep -v "__LEFTHAND__" > ${tmpFile}

# make sure empty recognition results get filled with [EMPTY] tags - so that the alignment can work out on all data.
cat ${tmpFile} | sed -e 's,\s*$,,'   | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' |sort -k1,1 -k3,3 > ${output}
rm ${tmpFile}
echo `date`
echo "Preprocess Finished."
