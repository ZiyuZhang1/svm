#!/bin/bash
echo "Watching for main_diffusion jobs of user ziyuzh..."
while pgrep -u ziyuzh -f "main_diffusion" > /dev/null; do
    sleep 300
done
echo "main_diffusion finished — starting next job."
bash /itf-fi-ml/shared/users/ziyuzh/svm/src/run_nn.sh
