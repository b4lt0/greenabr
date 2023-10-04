#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

FOLDER_NAME="$1"

AWK_CODE="{
                count[\$1]++
                total++
            }
            END {
                for (i=0; i<=5; i++) {
                    printf \"[%d]: %.2f%% \", i, (count[i]/total)*100
                }
                print \"\"
            }"

LOGS_PATH="./logs"
TARGET_PATH="$LOGS_PATH/$FOLDER_NAME"

if [ ! -d "$TARGET_PATH" ]; then
    echo "Error: Target path $TARGET_PATH does not exist."
    exit 1
fi

cd "$TARGET_PATH"
KNOWN_FILE_NAME="best_model.zip"
NEW_FILE_NAME="model.zip"
if [ -f "$KNOWN_FILE_NAME" ]; then
    mv "$KNOWN_FILE_NAME" "$NEW_FILE_NAME"
elif [ -f "$NEW_FILE_NAME" ]; then
    : # that's ok
else
    echo "Error: $KNOWN_FILE_NAME or $NEW_FILE_NAME do not exist in $TARGET_PATH."
    exit 1
fi

cp "$NEW_FILE_NAME" "../../"
cd "../../"

if [ -f "../venv/bin/python3.11" ]; then
  echo "[BITRATE LEVELS USAGE (tos, bbb, doc)]:"
  ../venv/bin/python3.11 evaluate.py tos $NEW_FILE_NAME 2>/dev/null | grep -E '^[0-5]$' | awk "$AWK_CODE"
  ../venv/bin/python3.11 evaluate.py bbb $NEW_FILE_NAME 2>/dev/null | grep -E '^[0-5]$' | awk "$AWK_CODE"
  ../venv/bin/python3.11 evaluate.py doc $NEW_FILE_NAME 2>/dev/null | grep -E '^[0-5]$' | awk "$AWK_CODE"
else
  echo "Error: the virtual environment with installed dependencies doesn't exist!"
  exit 1
fi


cd "./evaluation/rep_6"

python3.6 create_summary_results.py
python3.6 plot_graphs.py

cp ./plots/* "../../logs/$FOLDER_NAME"

echo "Evaluation completed successfully!"
echo "Check results at $TARGET_PATH!"