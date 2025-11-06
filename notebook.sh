# Starting the Jupyter notebook server
# Usage: . notebook.sh

( jupyter notebook --no-browser --port=9000 & echo $! > /tmp/jupyter_pid ) && trap "kill $(cat /tmp/jupyter_pid)" EXIT
